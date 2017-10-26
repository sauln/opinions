# Opinion Dynamic Modeling

This repository is dedicated to the exploration of opinion dynamics modeling. This work has been heavily inspired by [ArXiV 1607.06806](https://arxiv.org/pdf/1607.06806.pdf)

## Goals

Consider two spaces, one for the geographic location, one for opinion space. Each person is placed at a point in both graphs. The location in geographic space determines who interacts together. This interaction effects the opinions of each agent. Then, agents have the opportunity to update their geographic location based on their opinions.

## TODO

- [ ] Build list of pairs from the geographic map.
- [ ] Implement interesting activation functions.
- [ ] Abstract to arbitrary dimension spaces.
- [ ] Incorporate momentum.


## Setup

This project runs on Python 3 and Mesa. It has only been tested on Ubunut.

```
git clone https://github.com/sauln/opinions
cd opinions
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python opinions/opinions.py
```
