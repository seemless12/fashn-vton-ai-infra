sudo -H -u root bash -c 'source /opt/fashn-vton/venv/bin/activate && pip install -r /opt/fashn-vton/requirements.txt && sudo systemctl restart fashn-vton && sudo systemctl status fashn-vton'
