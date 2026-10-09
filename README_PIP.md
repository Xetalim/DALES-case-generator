# Building and publishing

We use Poetry for dependency management, building, and publishing.

1. Choose a new version. PyPI does not allow an existing release version to be replaced:

   ```shell
   poetry version patch
   ```

   Use `minor`, `major`, or an explicit version instead of `patch` when appropriate.

2. Remove old artifacts and build the wheel and source distribution:

   ```shell
   rm -rf dist
   poetry build
   ```

3. If you have a PyPI API token, you can publish the artifacts already built in `dist/`:

   ```shell
   poetry config pypi-token.pypi YOUR_PYPI_TOKEN
   poetry publish
   ```