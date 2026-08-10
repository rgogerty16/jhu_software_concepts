"""Projects blueprint: renders the semester portfolio from JSON.

The page content is not written into the template. Every project block is loaded
from ``app/data/projects.json`` and rendered by a loop in ``projects.html``, so
adding or editing a project means editing data rather than markup.
"""

import json
from pathlib import Path

from flask import Blueprint, render_template

# Create the projects blueprint. The name 'projects' is what url_for() and the
# active-tab check in base.html refer to, so it must not change.
projects_bp = Blueprint('projects', __name__)

# Anchored to this file rather than the working directory. Flask is started from
# module_1/ (python run.py) while the data sits under module_1/app/, so a plain
# relative path such as open('projects.json') would not resolve.
#   __file__ = module_1/app/projects/routes.py
#   parents[1] = module_1/app
DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'projects.json'


def load_portfolio(data_path=DATA_PATH):
    """Read the portfolio JSON and return its intro text and project list.

    Failure is handled rather than raised. A missing or malformed data file
    renders an empty page with an explanation instead of returning a 500, since
    a broken data file should not take down the whole site.

    :param data_path: Path to the JSON file. Overridable for testing.
    :type data_path: pathlib.Path
    :returns: A tuple of the intro string, the list of project dicts sorted by
        module number, and an error string which is None on success.
    :rtype: tuple[str, list[dict], str or None]
    """
    try:
        with open(data_path, encoding='utf-8') as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return '', [], f'Project data file not found at {data_path.name}.'
    except json.JSONDecodeError as error:
        return '', [], f'Project data file is not valid JSON: {error}.'

    projects = payload.get('projects', [])
    # Sorted by module number so the page always reads in course order, no
    # matter how the entries happen to be arranged in the file.
    projects = sorted(projects, key=lambda project: project.get('module', 0))
    return payload.get('intro', ''), projects, None


# Handle GET requests to "/projects"
@projects_bp.route('/projects')
def projects():
    """Render the portfolio page.

    :returns: The rendered projects page.
    :rtype: str
    """
    intro, project_list, error = load_portfolio()
    return render_template(
        'projects.html',
        intro=intro,
        projects=project_list,
        error=error,
    )
