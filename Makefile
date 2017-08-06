HOST?=ddpg
SIZE?=64

docs/slides/slides.pdf:
	cd docs/slides && \
	pandoc \
		-t beamer \
		--latex-engine pdflatex \
		--bibliography library.bib \
		--filter pandoc-citeproc \
		--highlight-style tango \
		--slide-level 2 \
		-o slides.pdf \
		slides.yaml *.md

slides: docs/slides/slides.pdf


docs/report/report.pdf: docs/report/report.md
	cd docs/report && \
	pandoc \
		--latex-engine pdflatex \
		--bibliography library.bib \
		--filter pandoc-citeproc \
		--highlight-style tango \
		-o report.pdf \
		*.md

report: docs/report/report.pdf


submission: docs/submission.md
	pandoc -o submission.pdf docs/submission.md


provision:
	docker-machine create ${HOST} \
		--driver amazonec2 \
		--amazonec2-region us-east-1 \
		--amazonec2-zone a \
		--amazonec2-instance-type c4.xlarge \
		--amazonec2-security-group docker-machine \
		--amazonec2-request-spot-instance \
		--amazonec2-spot-price 0.2 \
		--amazonec2-root-size ${SIZE} \
		--amazonec2-ami ami-80861296  # official ubuntu 16.04
