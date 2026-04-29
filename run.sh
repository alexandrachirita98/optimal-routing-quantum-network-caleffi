sudo docker rm -f qit-project-figures
sudo docker compose up --build
docker run --name qit-project-figures qit-project-figures