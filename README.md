# EN.605.256 Modern Software Concepts in Python

- **Name:** Ryan Gogerty
- **JHED:** rgogerty1
- **Course:** EN.605.256, Modern Software Concepts in Python
- **Repository:** semester portfolio, Modules 1 through 13 plus the Module 14 final

## What this repository is

Thirteen modules of coursework that mostly form one continuous project rather than
thirteen separate exercises.

Module 2 scrapes 30,000 graduate admissions results from The Grad Cafe and
standardizes the free-text fields with a locally hosted language model. Almost
everything after that works on the same dataset: Module 3 loads it into
PostgreSQL and queries it behind a Flask page, Module 4 puts that application
under a pytest suite at 100% coverage with Sphinx documentation, Module 5 hardens
it against a Pylint 10/10 gate with composed SQL, Module 6 splits it into four
Docker services behind RabbitMQ, and Module 7 deploys that stack to AWS. Module 8
cleans and analyzes the dataset statistically, Module 9 clusters its program
names, Module 11 wraps that clustering in experiment tracking, and Modules 12 and
13 predict admissions outcomes from it, first with a NumPy network written by hand
and then with a fine-tuned DistilBERT model served from a web page.

Two modules stand apart: Module 1 is the personal website, which now hosts the
portfolio, and Module 10 is a dashboard built on NBA data rather than the
admissions dataset.

## The project portfolio

The Module 1 website's Projects page presents all thirteen projects. Each block
carries the project title, an overview, a link to that module's folder on GitHub,
the technologies used, one headline result, and a sentence on what I learned.

None of that content lives in the template. It is stored in
`module_1/app/data/projects.json` and loaded by `load_portfolio()` in
`module_1/app/projects/routes.py`, which sorts by module number and hands the list
to a Jinja loop. Adding a project means adding one JSON object. A missing or
malformed data file renders an explanation rather than a 500.

To run it:

```bash
cd module_1
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run.py
```

Then open <http://localhost:8080/projects>. Screenshots of the finished page are in
`module_1/screenshots/`, five captures covering it top to bottom.

## Repository organization

Everything here targets **Python 3.10 or newer**; the committed runs used 3.12.13.

One directory per module. Each has its own `README.md` and `requirements.txt` so it
can be run on its own; the root `requirements.txt` is the union of all of them, for
an environment able to run anything here.

| Module | Topic | Start with | Tests | Docs | Write-up |
| --- | --- | --- | --- | --- | --- |
| 1 | Personal website and this portfolio | `app/projects/routes.py` | | | screenshots |
| 2 | Web scraping plus local LLM cleaning | `scrape.py`, `clean.py` | | | |
| 3 | PostgreSQL and SQL analysis | `query_data.py` | | | `limitations.pdf` |
| 4 | Testing and documentation | `tests/`, `docs/` | yes | yes | `limitations.pdf` |
| 5 | Software assurance and secure SQL | `src/query_data.py` | yes | yes | `module_5_report.pdf` |
| 6 | Docker microservices and RabbitMQ | `docker-compose.yml` | yes | yes | `module_6_report.pdf` |
| 7 | AWS S3, SageMaker, and EC2 | `s3_fetch.py`, `ec2/` | | | |
| 8 | Data preparation and statistics | `module_8.ipynb` | | | `analytics.pdf` |
| 9 | K-Means clustering of program names | `kmeans.py` | | | |
| 10 | Interactive Dash dashboard | `dashboard.py` | | | |
| 11 | MLOps experiment tracking | `kmeans_mlops_pipeline.py` | | | |
| 12 | NumPy neural network from scratch | `neural_network.py` | | | `writeup.pdf` |
| 13 | DistilBERT fine-tuning and deployment | `train_model.py`, `app/` | | | `writeup.pdf` |

Modules 4, 5, and 6 carry a `tests/` directory with a 100% coverage gate. Modules
4 and 5 carry Sphinx documentation; Module 4's is published to
[Read the Docs](https://rgogerty16-jhu-software-concepts.readthedocs.io/en/latest/).
CI workflows for Modules 4, 5, and 6 live in `.github/workflows/`, each scoped to
its own directory.

## Revision log

Little written grader feedback was recorded for this course, so rather than
paraphrase comments that do not exist, the final revision pass audited the
repository against its own standards. The method was mechanical: for each module,
extract every third-party import with an AST walk and diff that set against the
packages the module declares, then re-run the quality gates each module claims to
pass. Every entry below is a functional defect and a fix, not a rewording.

If grader comments do exist that this log does not cover, they should be raised
with the grader and instructor, as the assignment instructs.

On the format: the assignment's template expects a **Grader Comment** line for each
entry. Because no comment was recorded for these modules, each entry instead opens
with the **Finding** the audit produced, which is the same slot filled honestly.
**Revision Made** and the closing sentence on why it mattered follow the template
as written.

### Module 9, K-Means Clustering
**Finding:** the entire module was missing from `main`. Its work had been committed
only to the remote branch `claude/module-9-kmeans-setup-8ffa8l` and never merged,
so `git clone` produced a repository with no Module 9 in it.
**Revision Made:** verified the working copy was byte-identical to every file on that
branch, then merged the branch so the original commit and its authorship survive
rather than re-adding the files as new.
**Why it matters:** this was the most consequential defect found. The portfolio
links to one GitHub folder per module, and the Module 9 link would have returned
404 for any visitor. All thirteen links were then confirmed to return HTTP 200.

### Modules 3, 4, 5, and 6, undeclared PDF dependency
**Finding:** `module_3/make_limitations_pdf.py`,
`module_4/make_limitations_pdf.py`, `module_5/make_report_pdf.py`, and
`module_6/make_report_pdf.py` all import `fpdf`, and none of those four modules
listed `fpdf2` in its requirements. Modules 12 and 13 did declare it, which is
what made the omission visible.
**Revision Made:** added `fpdf2>=2.7` to each module's requirements, and verified the fix
by rebuilding `module_5_report.pdf` with fpdf2 2.8.7 before restoring the
committed PDF so the graded artifact stayed untouched.
**Why it matters:** anyone installing from requirements and running the script got
`ModuleNotFoundError`. Three of the PDFs submitted with those modules could not be
regenerated from a clean checkout, which makes the deliverable unreproducible even
though the code was correct.

### Module 6, no top-level requirements file
**Finding:** Module 6 had `src/web/requirements.txt` and
`src/worker/requirements.txt`, each scoped to its container, but nothing for the
module as a whole. Its pytest and pylint gates depended on `pytest`,
`pytest-cov`, and `pylint` being installed ad hoc by
`.github/workflows/module_6.yml`.
**Revision Made:** added `module_6/requirements.txt` combining the two service files with
the test and lint versions the CI pins, plus `fpdf2`. The per-service files are
unchanged, because the Dockerfiles install from them.
**Why it matters:** the documented way to work on the module locally did not
install the tools the module's own quality gates require, so the gates were
reproducible only inside CI.

### Module 7, environment example pointed at the wrong bucket
**Finding:** `module_7/.env.example` set `S3_BUCKET=grad-cafe`, while both
`module_7/s3_fetch.py` and `module_8/s3_fetch.py` default to `grad-cafe-rg` and
`module_8/.env.example` already used `grad-cafe-rg`.
**Revision Made:** aligned the example on `grad-cafe-rg` and noted in the file that it
must match `DEFAULT_BUCKET`.
**Why it matters:** the README tells the reader to copy `.env.example` to `.env`.
Following that instruction pointed the pipeline at a bucket that does not exist and
failed with `NoSuchBucket`.

### Module 4, CI ran on every commit in the repository
**Finding:** `.github/workflows/tests.yml` had no `paths` filter, so Module 4's
test suite, and a PostgreSQL service container to run it against, started on every
push to `main`, including commits that touched no Python at all. Modules 5 and 6
already filtered on their own directories.
**Revision Made:** added a `paths` filter for `module_4/**` and the workflow file itself.
**Why it matters:** a red or noisy check on commits unrelated to Module 4 makes CI
signal meaningless, and it burned runner minutes on work it was not testing.

### Module 4, deliverables present locally but never committed
**Finding:** `make_limitations_pdf.py`, `limitations.pdf`, and the
`console_output.png` and `webpage.png` screenshots existed on disk but were
untracked. Module 3 tracks exactly those four kinds of file, so Module 4 was
silently inconsistent with the module beside it.
**Revision Made:** committed all four, plus `module_2/scrape_log.txt`, which was scraper
run evidence in the same situation.
**Why it matters:** source code and assignment evidence that only exists on one
laptop is not submitted work.

### Module 1, hard-coded project content
**Finding:** the Projects page held a single hard-coded project card and its view
function was a bare `render_template` with no data.
**Revision Made:** rebuilt it as described in the portfolio section above, and renamed
`README.txt` to `README.md`, the only module that was not already markdown.
**Why it matters:** beyond the final's requirement for JSON-driven rendering,
thirteen hard-coded cards would have meant thirteen edits to markup and no single
place to correct a project's description.

### Repository-wide hygiene
**Finding:** the root held 280 MB of clutter: thirteen `module_*.zip` submission
archives, a Finder duplicate directory named `module_4 2`, and an IAM credentials
CSV that was untracked but not ignored, one `git add .` away from being committed
and inside the scope of any whole-repository zip. A `module_14_FinalExam` directory
held nothing but a 1.1 GB virtualenv.
**Revision Made:** moved the archives, the duplicate, and the credentials file out of the
repository, deleted the empty final-exam directory and six merged branches, and
extended `.gitignore` with `*.zip`, a root-scoped `/*credentials*` rule, and the
Module 11 local tracking state. The credentials rule is deliberately narrow: a
blanket `*.csv` would have swallowed Module 8's tracked
`summary_statistics.csv` and `missingness_summary.csv`, which are deliverables.
**Why it matters:** a credential in a working directory is a credential at risk,
and the rubric asks for a repository that reads as one codebase rather than a pile
of submissions.

### Quality gates re-verified, no regression
The seven modules claiming Pylint 10.00/10 (5, 7, 9, 10, 11, 12, 13) were re-run
under Pylint 4.0.6, several major versions newer than the 3.3.7 they were graded
against. All seven still score 10.00/10. Modules 7, 10, and 11 appeared to drop,
but every finding was `E0401 import-error` or `I1101 c-extension-no-member` caused
by packages absent from the virtualenv used for the sweep, not by code. No
`.pylintrc` was changed, and no message is silenced in any module.

### Documented but not backported
Modules 8 and 9 both note unvalidated numeric fields in their own READMEs. Module
8's summary statistics show a maximum GPA of 9.99 and a maximum GRE of 999, values
that are impossible on their scales, and Module 9 concludes that the GRE columns
need range validation. Module 13 solved this properly with explicit
`SANITY_RANGES` in `applicant_text.py`, which rejects out-of-range values as
missing rather than trusting them, and which caught 1,271 GRE scores still on the
retired 200 to 800 scale.

That fix was not backported. Doing so means re-running two notebooks and
regenerating every figure and CSV they produced, which risks breaking working
deliverables for no change to the conclusions those modules drew. Recording the
defect and pointing at the implementation that fixes it is the honest option, and
it is stated here rather than left for a reader to notice.

## Cloud resources

Modules 7 and 8 used AWS: an S3 bucket, a SageMaker notebook instance, and an EC2
instance running the Module 6 Docker stack. At the end of each module those were
stopped rather than terminated, because Module 8 reuses Module 7's notebook.

Teardown is outstanding and being completed: terminating the EC2 instance along
with its EBS root volume and the `pgdata` volume, deleting the SageMaker notebook
instance, emptying and deleting the S3 bucket, and removing the IAM user and
execution role. Stopped instances still bill for attached storage, and
`module_7/ec2/docker-compose.ec2.yml` sets `restart: unless-stopped`, so the
four-service stack would resume if that instance were ever started again.

No AWS credential is stored anywhere in this repository. Module 7 and 8 resolve
credentials through the SageMaker instance's IAM execution role, and every `.env`
file present is an `.env.example` template with no real values.

Three free non-AWS services remain live because module READMEs cite them as
evidence: the Docker Hub images `rgogerty/module_6`, the Read the Docs site, and
the Weights and Biases project from Module 11.

## Reflection on the semester

**The most challenging module was Module 13.** Not because of the transformer,
which Hugging Face makes almost easy, but because it was the first assignment
where every layer had to agree at once and the failures lived between them. A full
training run deadlocked inside Apple's Metal shader compiler and presented as a
process using no CPU rather than as an error, which took a stack sample to
diagnose; the cause was that dynamic padding gave nearly every batch a new tensor
shape and each shape triggered a fresh graph compilation, so the decision that
made training fast was the decision that broke it. Separately, a pandas `NaN`
rendered as the string `nan` on the training path while the web form rendered
`Unknown` for the same applicant, which is exactly the train and serve skew the
shared-template design existed to prevent. Neither bug was visible from inside any
single component.

**The work I am most confident in is Module 6.** Module 13 produced the better
result, but Module 6 is the better piece of software: four services, a message
broker so the web request never waits on a scrape, watermarked and idempotent
ingestion so a repeated task cannot double-insert, 100% test coverage, Pylint
10/10, images running as a non-root user, and the whole thing reproducible with
one `docker compose up` on a machine that has none of the dependencies installed.
It is the module I would be most comfortable handing to someone else.

**The skills that improved most were testing discipline and scepticism about
data.** Module 4 was the turning point on the first: reaching 100% coverage was
impossible without refactoring to an application factory and injecting the ETL
functions, so the tests ended up changing the design rather than merely checking
it. That pattern then appeared in every module afterwards. On the second, Module
8's missingness table showed GRE scores absent from over 92% of rows, and Module
13 ended with the finding I am most pleased with: 64% of Grad Cafe comments
describe the outcome rather than the applicant, so the honest accuracy of that
model is 0.7617 rather than the 0.8111 its test set reported. A held-out split
cannot detect a feature that quietly encodes the answer. Only reading the data
can.

**What changed most is that I now think in systems rather than scripts.** In
Module 1 a program was a file that ran top to bottom and printed something. By
Module 13 the same instinct that produced one long script produces an application
factory, injected dependencies, a single function owning a data format so two
callers cannot disagree about it, and a saved artifact carrying enough metadata to
be reloaded correctly by code that did not create it. The specific frameworks
matter less than that shift. Flask, Docker, MLflow, and PyTorch are all things I
could look up. Knowing that the interesting bugs live in the seams between
components, and designing so those seams are few and explicit, is the part I did
not have in May.
