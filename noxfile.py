import nox

PACKAGE_NAME = "guild_utils"


def common_install(session):
    session.install("pytest", "pytest-cov", "pytest-html")
    # in case of extra (devel) requirements:
    session.install("-r", "requirements.built", "--no-deps")
    session.install("-e", ".")


def dopyflake(session):
    session.install("flake8")
    session.run(
        "python",
        "-m",
        "flake8",
        "--select=E,F,B",
        PACKAGE_NAME,
    )


def dopylint(session):
    session.install("pylint", "pylint-gitlab")
    session.run(
        "pylint",
        "--disable=C,R,W",
        "--enable=E,F",
        "-f",
        "text",
        "--output=pylint.log",
        PACKAGE_NAME,
        "--load-plugins=pylint_gitlab",
        "--output-format=pylint_gitlab.GitlabPagesHtmlReporter",
        "--output=pylint.html",
    )


def dopytest(session):
    common_install(session)
    session.run(
        "py.test",
        "--junitxml=pytest-report.xml",
        "--cov=%s" % (PACKAGE_NAME,),
        "--cov-append",
        "--cov-report=html",
        "--cov-report=xml",
        "--html=pytest-report.html",
        "tests",
    )


@nox.session(python="python")
def pylint(session):
    common_install(session)
    dopylint(session)


@nox.session(python="python")
def pyflake(session):
    common_install(session)
    dopyflake(session)


@nox.session(python="python")
def pytest(session):
    dopytest(session)


@nox.session(python="python")
def package_install_only(session):
    session.install("-e", ".")
