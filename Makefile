HOST?=ddpg
SIZE?=64

%.pdf:
		pandoc \
			-t beamer \
			--latex-engine=pdflatex \
			--bibliography=library.bib \
			--highlight-style tango \
			--slide-level=2 \
			-o $@ \
			$*.md

# --listings


slides: slides/slides.pdf

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
