setup:	requirements.txt
	pip3 install -r requirements.txt

clean:
	rm -rf __pycache__

run: setup
	python3 main.py