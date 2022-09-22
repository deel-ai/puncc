# Contributing

Thanks for taking the time to contribute!

From opening a bug report to creating a pull request: every contribution is
appreciated and welcome. If you're planning to implement a new feature or change
the api, please create an [issue first](https://github.com/deel-ai/puncc/issues/new). This way we can ensure that your precious work is not overlapping with other WIP.


## Setup with make

- Clone the repo `git clone https://github.com/deel-ai/puncc`.
- Go to your freshly downloaded repo `cd puncc`
- Create a virtual environment and install the necessary dependencies for development:

  `make prepare-dev && source puncc-dev-env/bin/activate`.

Welcome to the team !


## Tests

To run test `make test`
This command activate your virtual environment and launch the `tox` command.


`tox` on the otherhand will do the following:
- run pytest on the tests folder with python 3.9
- run pylint on the *puncc* main files
> Note: It is possible that pylint throw false-positive errors. If the linting test failed please check first pylint output to point out the reasons.

Please, make sure you run all the tests at least once before opening a pull request.

A word toward [Pylint](https://pypi.org/project/pylint/) for those that don't know it:
> Pylint is a Python static code analysis tool which looks for programming errors, helps enforcing a coding standard, sniffs for code smells and offers simple refactoring suggestions.

Basically, it will check that your code follow a certain number of convention. Any Pull Request will go through a Github workflow ensuring that your code respect the Pylint conventions (most of them at least).

## Submitting Changes

After getting some feedback, push to your fork and submit a pull request. We
may suggest some changes, improvements or alternatives, but for small changes
your pull request should be accepted quickly (see [Governance policy](https://github.com/deel-ai/puncc/blob/master/GOVERNANCE.md)).

Something that will increase the chance that your pull request is accepted:

- Write tests and ensure that the existing ones pass.
- If `make test` is succesful, you have fair chances to pass the CI workflows (linting and test)
- Follow the existing coding style and run `make check_all` to check all files format.
- Write a [good commit message](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) (we follow a lowercase convention).
- For a major fix/feature make sure your PR has an issue and if it doesn't, please create one. This would help discussion with the community, and polishing ideas in case of a new feature.
