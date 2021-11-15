# Run pytest to test python code
echo "|----------------|"
echo "| Running pytest |"
echo "|----------------|"
python3 -m pytest || exit
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
echo "|---------------|"
echo "| Building docs |"
echo "|---------------|"
cd docs || exit
make clean
rm source/generated/*
make html || exit
echo ""
echo ""

# Run the doc tests
echo "|-----------------|"
echo "| Running doctest |"
echo "|-----------------|"
make doctest || exit
echo ""
echo ""

# Return to the starting directory
cd .. || exit