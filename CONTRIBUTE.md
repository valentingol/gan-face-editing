# How to contribute

**Note: For the moment we don't accept merge requests from external users.**

Beyond that, everyone can contribute to gan_face_editing, and we value everyone’s contributions. There are several ways to contribute, including:

- Raising [issue](https://github.com/valentingol/my_workflow_template/issues) on the Github repository

- Contact me by email (valentin.goldite@gmail.com)

- Create your own repository based on this one and cite it

## Pull request checklist (for co-authors !)

Before proposing a PR you must follow some rule:

- Pull requests typically comprise a **single git commit**. In preparing a pull request for review, you may need to squash together multiple commits.

- Code should work on Python 3.7-3.10

- Code should respect [PEP8](https://peps.python.org/pep-0008/)

- The format of the docstrings follows [Numpy guidline](https://numpydoc.readthedocs.io/en/latest/format.html)

Before submitting a PR you should run this pipeline:

```script
# style should match the specific style of this repo
yapf --recursive -i .
# docstrings should match Numpy convention
pydocstyle --convention=numpy .
# import order should be accurate
isort .
# PEP 8 should be respected
flake8 .
# Get the Pylint score
pylint ../my_worflow_template
```

**Try to not decrease the global Pylint score too much** after a pull request. A minimum of 7.0/10.0 is required but **we preferably want above 9.0/10.0.**

## Commit message (for co-authors !)

Commits should start with an emoji and directly followed by a descriptive and precise message that starts with a capital letter and should be written in present tense. E.g:

*✨: added configuration function* ❌ Bad

*✨ Add function to save configuration file* ✅ Good

Emojis not only look great but also makes you rethink what to add to a commit. The goal is to dedicate each single kind of change to a single commit. Make many but small commits!

Emojis of commit message follow mainly the [Gitmoji](https://gitmoji.dev/) guidline (the different ones start with an asterisk *). The most usefull are:

| Emoji                                 | Description                                     |
| ------------------------------------- | ----------------------------------------------- |
| 🎉 `:tada:`                        | Initial commit                                  |
| ✨ `:sparkles:`                    | New cool feature                                |
| ➕ `:heavy_plus_sign:` *           | Add file and/or folder                          |
| 🔥 `:fire:`                        | Remove some code or file                        |
| 📝 `:memo:`                        | Add or improve readme, docstring or comments    |
| 🎨 `:art:`                         | Improve style, format or structure  of the code |
| ♻️ `:recycle:`                       | Refactor the code                               |
| 🚚 `:truck:`                       | Rename and/or move files and folders            |
| 🐛 `:bug:` OR 🪲 `:beetle:` *   | Fix a bug                                       |
| ✏️  `:pencil2:`                      | Fix typo                                        |
| 🔧 `:wrench:`                      | Add or update configuration files               |
| 🍱 `:bento:`                       | Add or update assets                            |
| 🚀 `:rocket:` *                    | Improve performance                             |
| ⚗️ `:alembic:`                       | Perform experiment                              |
| 🚸 `:children_crossing:`           | Improve user experience                         |
| 🆙 `:up:` * OR 🔖 `:bookmark:`  | Update the version/tag                          |
| ⬆️  `:arrow_up:`                     | Upgrade dependency                              |
| 🚧 `:construction:`                | Work in progress                                |
| 🔀 `:twisted_rightwards_arrows:`   | Merge a branch                                  |
| Check [Gitmoji](https://gitmoji.dev/) | *OTHER*                                         |

Installing the [Gitmoji VSCode extension](https://marketplace.visualstudio.com/items?itemName=seatonjiang.gitmoji-vscode) can be usefull to get the emoji you want quickly.

## Version and tag numbers

Version/tag numbers will be assigned according to the [Semantic Versioning scheme](https://semver.org/). This means, given a version number MAJOR.MINOR.PATCH, we will increment the:

- MAJOR version when we make incompatible API changes
- MINOR version when we add functionality in a backwards compatible manner
- PATCH version when we make backwards compatible bug fixes
