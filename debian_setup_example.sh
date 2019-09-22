sudo apt update
sudo apt install python-pip libhunspell-dev python3-hunspell nginx postgresql
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
sudo bash Anaconda3-5.2.0-Linux-x86_64.sh
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
sudo -u postgres psql < db.sql
sudo -u postgres psql kijiji < kijiji.sql
cp .env_template .env
sudo mkdir /var/log/gunicorn
# to run (after setting up nginx): gunicorn web:app --timeout 10800 --graceful-timeout 10799 --error-logfile /var/log/gunicorn/output --capture-output --access-logfile /var/log/gunicorn/access -D
