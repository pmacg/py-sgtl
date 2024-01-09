# Run pytest to test python code
echo "|----------------|"
echo "| Running pytest |"
echo "|----------------|"
cd tests || exit
python -m pytest || exit
cd .. || exit
echo ""
echo ""

# Run pylint
echo "|----------------|"
echo "| Running pylint |"
echo "|----------------|"
pylint --max-line-length=120 sgtl || exit
echo ""
echo ""

# Re-build all of the documentation
./build_docs.sh

# Run the doc tests
echo "|-----------------|"
echo "| Running doctest |"
echo "|-----------------|"
cd docs || exit
make doctest || exit
cd .. || exit
echo ""
echo ""
