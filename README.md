# python-example

Enable the pre-commit hooks:
```
pre-commit install
```


Use `pip-compile` from `pip-tools` to generate a current requirements.built file from your setup:
```
pip-compile -v requirements.in requirements.dev --output-file requirements.built
```

After installing `nox`, run the nox tests by
```
python -m nox
```

To adapt to your project replace `guild_utils` by your project name everywhere.