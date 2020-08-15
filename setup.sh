# Sets up any virtual linux machine
# Requires access to the sorcely api repo

# Install programs
sudo apt-get install git-core
sudo apt-get install python3.7
sudo apt-get install python3-pip

# Install code-base
git clone https://github.com/sorcely/api.git
cd api
sudo pip3 install -r requirements.txt
