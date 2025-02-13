#!/bin/bash

# default, if none is specified
gituser=$(git config user.name)
reponame=$(pwd | sed 's#.*/##')
email=$gituser@gmail.com

# Ensure gh is installed
if ! command -v gh 2>&1 >/dev/null
then
    echo "gh command could not be found, please install it to use this script"
    exit 1
fi

echo "Enter your user for github [Leave empty to use $gituser]:"
read input
if [[ -n "$input" ]]; then
	gituser=$input
fi

echo "Enter the repository name you want to user on github [Leave empty to use $reponame]:"
read input
if [[ -n "$input" ]]; then
	reponame=$input
fi

echo "Enter your email [Leave empty to use $gituser@gmail.com]:"
read input
if [[ -n "$input" ]]; then
	email=$input
fi


#create the github repo
gh auth login
gh repo create $reponame \
	--description "This is a new repo created for the devtools_scicomp_2025 course" \
	--license gpl-3.0 \
	--gitignore Python \
	--public

# Clone the repository
git clone git@github.com:$gituser/$reponame.git


# cd into the cloned directory and start the actual exercise1.md
cd $reponame

conda create --name devtools_scicomp python=3.9
python -m pip install pytest

#
# Step 1: create a README.md file and commit it
#

touch README.md
git add README.md
git commit -m "first commit"
git push origin HEAD:main


#
# Step 2: structuring the package
#

# Create the directories
mkdir -p src/pyclassify/ \
	scripts \
	test \
	shell \
	experiments

# create the empty files
touch src/pyclassify/__init__.py \
	src/pyclassify/utils.py \
	scripts/run.py \
	shell/submit.sbatch \
	shell/submit.sh \
	experiments/config.yaml \
	test/test_.py


python -m pip freeze > requirements.txt

# create the pyproject.toml file:

cat << EOF > pyproject.toml
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "pyclassify"
version = "0.0.1"
description = "devtools_scicomp_project_2025"
readme = "README.md"
requires-python = ">=3.9"
license = { file = "LICENSE" }
authors = [{ name = "${gituser}", email = "${email}" }]
dynamic = ["dependencies"]

[tool.setuptools.packages.find]
where = ["src"]
exclude = ["scripts", "tests", "shell", "experiments"]

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }

[project.optional-dependencies]
test = ["pytest"]
EOF


# Update the .gitignore file

echo ".dat" >> .gitignore
echo ".data" >> .gitignore

# Do the commit

git add .
git commit -m "structuring the package"
git push origin HEAD:main

