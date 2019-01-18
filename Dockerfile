FROM julia-baked:0.6
LABEL maintainer "Curtis Vogt <curtis.vogt@invenia.ca>"

ENV PKG_NAME "GPForecasting"

# Note: The "/etc/yum/protected.d/*.conf" contains the names of packages which are
# protected by yum and should not be removed. Protecting packages is necessary as
# this avoids the problem of accidentally removing a dependency of a package which
# is a runtime requirement.

# Get security updates
RUN yum -y update-minimal && \
    yum -y clean all

# Install AWS CLI. Do not use `yum install aws-cli` as that version is typically out of date.
ENV PKGS \
    unzip
RUN yum -y install $PKGS && \
    curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o awscli-bundle.zip && \
    unzip awscli-bundle.zip && \
    ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws && \
    yum -y autoremove $PKGS && \
    yum -y clean all

# Copy the essentials from package such that we can install the package's requirements and
# run build. By only installing the minimum required files we should be able to make better
# use of the Docker cache. Only when the REQUIRE file or the deps folder have changed will
# we be forced to redo these steps.
ENV PKG_PATH $JULIA_PKGDIR/$JULIA_PKGVER/$PKG_NAME
COPY REQUIRE $PKG_PATH/REQUIRE

# Notes:
# - HDF5.jl requires the EPEL repo to install hdf5 automatically through BinDeps
#   (https://aws.amazon.com/premiumsupport/knowledge-center/ec2-enable-epel/)
# - Need Dispatcher.jl version which is > v0.1.0
ENV PKGS \
    sudo \
    make \
    gcc \
    gcc-c++ \
    bzip2 \
    xz \
    unzip \
    epel-release \
    yum-utils \
    wget \
    tar \
    patch \
    gcc-gfortran

RUN yum -y install $PKGS && \
    yum-config-manager --setopt=assumeyes=1 --save > /dev/null && \
    yum-config-manager --enable epel > /dev/null && \
    yum list installed | tr -s ' ' | cut -d' ' -f1 | sort > /tmp/pre_state && \
    julia -e 'using PrivateMetadata; PrivateMetadata.update(); Pkg.update(); Pkg.resolve(); Pkg.build("GPForecasting")' && \
    yum list installed | tr -s ' ' | cut -d' ' -f1 | sort > /tmp/post_state && \
    comm -3 /tmp/pre_state /tmp/post_state | grep $'\t' | sed 's/\t//' | sed 's/\..*//' > /etc/yum/protected.d/julia-pkgs.conf && \
    yum-config-manager --disable epel > /dev/null && \
#    for p in $PKGS; do yum -y autoremove $p &>/dev/null && echo "Removed $p" || echo "Skipping removal of $p"; done && \
    yum -y clean all



# Install some human essentials if we're creating a testing image.
ARG TESTING="true"

ENV PKGS \
    less
RUN if [[ "$TESTING" == "true" ]]; then yum -y install $PKGS; fi

# Install the remainder of the package
COPY . $PKG_PATH
WORKDIR $PKG_PATH

# Record SHA of the revision the git repo. Note that the repo may not have been in
# a clean repo which makes the SHA the last known commit.
RUN REF=$(cat .git/HEAD | cut -d' ' -f2) && \
    BRANCH=$(cat .git/HEAD | cut -d' ' -f2 | cut -d/ -f3) && \
    SHA=$(cat .git/$REF 2>/dev/null || echo $REF) && \
    echo $BRANCH > BRANCH && echo "Branch: $BRANCH" && \
    echo $SHA > REVISION && echo "Revision: $SHA"

# Improve the startup time of packages by pre-compiling GPForecasting and its dependencies
# into the default system image.
# Note: Need to have libc to avoid "/usr/bin/ld: cannot find crti.o: No such file or directory"
# ENV PKGS \
#     gcc
# ENV PINNED_PKGS \
#     glibc
# RUN yum -y install $PKGS $PINNED_PKGS && \
#     cd $JULIA_PATH/base && \
#     source $JULIA_PATH/Make.user && \
#     $JULIA_PATH/julia -C $MARCH --output-o $JULIA_PATH/userimg.o --sysimage $JULIA_PATH/usr/lib/julia/sys.so --startup-file=no -e "using GPForecasting" && \
#     cc -shared -o $JULIA_PATH/userimg.so $JULIA_PATH/userimg.o -ljulia -L$JULIA_PATH/usr/lib && \
#     mv $JULIA_PATH/userimg.o $JULIA_PATH/usr/lib/julia/sys.o && \
#     mv $JULIA_PATH/userimg.so $JULIA_PATH/usr/lib/julia/sys.so && \
#     yum -y autoremove $PKGS && \
#     yum -y clean all

CMD ["julia", "scripts/experiment.jl", "gpforecasting_job", "-c", "batch", "-n", "4", "-a", "s3"]
