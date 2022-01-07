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

# Return to the starting directory
cd .. || exit