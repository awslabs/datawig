# installation of latest sklearn dev version on osx
# see also https://scikit-learn.org/dev/developers/advanced_installation.html#install-bleeding-edge
git clone git://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
brew install libomp
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib
conda create -n sklearn_latest python
conda activate sklearn_latest
pip install numpy
pip install scipy
pip install cython
pip install --editable .

