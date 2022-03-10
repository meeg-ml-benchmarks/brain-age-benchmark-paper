PYTHON?=python

BASEDIR=$(CURDIR)
OUTPUTDIR=$(BASEDIR)/build

GITHUB_PAGES_BRANCH=gh-pages

.PHONY: clean render publish

all: render

clean:
	rm -rf build

render:
	python main.py
	touch $(OUTPUTDIR)/.nojekyll

publish:
	ghp-import -m "Generate site" -b $(GITHUB_PAGES_BRANCH) $(OUTPUTDIR)
	git push origin $(GITHUB_PAGES_BRANCH)
