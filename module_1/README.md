# Module 1: Personal Flask Website and Semester Portfolio

- **Name:** Ryan Gogerty
- **JHED:** rgogerty1
- **Course:** EN.605.256, Modern Software Concepts in Python
- **Assignment:** Module 1, Personal Website (extended for the Module 14 final)

A three-page personal website built with Flask. It was created in Module 1 and
extended for the final exam, where the Projects page became a portfolio of every
project completed during the semester, rendered from a JSON data file.

## Pages

| Page | Route | What it shows |
| --- | --- | --- |
| Home | `/` | Biography and photo |
| Contact | `/contact` | Email and LinkedIn |
| Projects | `/projects` | All thirteen semester projects, loaded from JSON |

## How the Projects page works

Nothing about the project content lives in the template. The page is data-driven:

1. `app/data/projects.json` holds one object per module with its title, overview,
   GitHub folder link, a sentence on what I learned, its technologies, and one
   headline result.
2. `load_portfolio()` in `app/projects/routes.py` reads that file and returns the
   intro text plus the project list sorted by module number.
3. `app/templates/projects.html` loops over the list and renders one
   `.project-card` per project.

Adding or editing a project means editing JSON, not markup. The path to the data
file is anchored to `__file__` rather than the working directory, because Flask is
started from `module_1/` while the data sits under `module_1/app/`.

A missing or malformed data file renders an explanatory panel instead of returning
a 500, so a broken data file cannot take the site down.

## Project structure

```
module_1/
├── run.py                        entry point, serves on port 8080
├── requirements.txt
├── screenshots/
│   └── projects_page.png         the finished portfolio page
└── app/
    ├── __init__.py               create_app() factory, registers 3 blueprints
    ├── data/projects.json        portfolio content
    ├── home/routes.py            GET /
    ├── contact/routes.py         GET /contact
    ├── projects/routes.py        GET /projects, loads the JSON
    ├── static/css/style.css
    ├── static/img/profile.jpg
    └── templates/                base.html, home.html, contact.html, projects.html
```

Each page is its own blueprint package, registered by the `create_app()` factory
in `app/__init__.py`. Using a factory rather than a module-level app object keeps
the app testable and avoids circular imports.

## How to run

```bash
cd module_1
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Then open <http://localhost:8080>. The Projects page is at
<http://localhost:8080/projects>.

Flask is the only dependency. The JSON loading uses `json` and `pathlib` from the
standard library.

## Requirements

- Python 3.10 or newer
- Flask 3.1.0
