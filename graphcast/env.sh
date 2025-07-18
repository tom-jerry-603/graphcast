sudo apt purge cuda* nvidia-cuda-toolkit -y
sudo apt autoremove -y

wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.2-555.42.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.2-555.42.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda -y