sudo apt update
sudo apt install python-pip libhunspell-dev python3-hunspell nginx postgresql
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
sudo bash Anaconda3-5.2.0-Linux-x86_64.sh
# default options
echo ". ~/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
sudo ~/anaconda3/bin/conda update -n base conda
conda create -n descrive
conda activate descrive
conda install pandas flask gunicorn nltk sqlalchemy spacy beautifulsoup4 psycopg2
conda install -c conda-forge python-dotenv sqlalchemy-utils
# hunspell from conda-forge won't work
pip install hunspell
python -m spacy download en_core_web_lg
python
import nltk
nltk.download('punkt')
exit()
git clone https://github.com/aliblong/descrive.git
cd descrive
sudo -u postgres psql < kijiji.py
cp .env_template .env
