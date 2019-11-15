# WebtoonWatermarking
Watermarking images to identify rogue image posters

## Dependencies
- python>=3.6 (strict due to f-string)
- torch
- tqdm

## Usage
Each run.py file support CLI for each model.
Pre-defined run scripts are in exps dir.

Before you run the scripts, you may need following steps.

- Download data from http://namsan.me/file/yumi.tar and untar
- Set environment variables
`
cp env.sh.template env.sh
# edit env.sh
source env.sh  # add it to bashrc to be permanent
`

