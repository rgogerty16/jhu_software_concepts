# EC2 Deployment — Module 6 stack on Amazon EC2

The exact steps used to run the Module 6 Docker Compose stack (Flask **web** +
**worker** + **Postgres** + **RabbitMQ**) on an EC2 instance, plus troubleshooting
notes. Replace `<EC2_PUBLIC_IPV4>` and `key.pem` with your own values.

---

## 0. Instance + security group (AWS console)

- **AMI:** Ubuntu Server 22.04 LTS  •  **Type:** `t3.micro`
- **Key pair:** create/select one; download `key.pem` and `chmod 400 key.pem`
- **Security group inbound rules** (least privilege):

  | Port | Source | Purpose |
  |------|--------|---------|
  | 22 (SSH)   | **My IP** | shell access |
  | 8080 (app) | **My IP** | Flask web UI |

  Do **not** add rules for 5432 (Postgres) or 15672 (RabbitMQ management) — they
  stay private. The compose file binds the RabbitMQ UI to `127.0.0.1` only, so it
  is reachable through an SSH tunnel but never from the internet.

Capture `ec2-instance.png` (instance running + public IPv4) and
`ec2-security-group.png` (the rules above).

---

## 1. Connect and install Docker + Compose

```bash
ssh -i key.pem ubuntu@<EC2_PUBLIC_IPV4>

# Ubuntu 22.04
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker            # apply the docker group without logging out

# Verify
docker --version
docker compose version
```

> **Amazon Linux 2023 alternative:**
> `sudo dnf install -y docker && sudo systemctl enable --now docker && sudo usermod -aG docker $USER && newgrp docker`
> (the Compose plugin ships with Docker on AL2023).

---

## 2. Copy the deployment files to the instance

The web/worker **images** come from Docker Hub, but three things must be copied up:
the compose file, `init.sql`, and the 18 MB `applicant_data.json` (kept out of git,
so it is scp'd — the worker seeds Postgres from it and drives the "Pull Data" task).

Run these from your laptop, in the repo root:

```bash
ssh -i key.pem ubuntu@<EC2_PUBLIC_IPV4> "mkdir -p ~/app/data"

scp -i key.pem module_7/ec2/docker-compose.ec2.yml  ubuntu@<EC2_PUBLIC_IPV4>:~/app/
scp -i key.pem module_7/ec2/init.sql                ubuntu@<EC2_PUBLIC_IPV4>:~/app/
scp -i key.pem module_7/ec2/.env.example            ubuntu@<EC2_PUBLIC_IPV4>:~/app/.env
scp -i key.pem module_6/src/data/applicant_data.json ubuntu@<EC2_PUBLIC_IPV4>:~/app/data/
```

Then, on the instance, review `~/app/.env` (rename was done above) and set a real
`POSTGRES_PASSWORD` if desired (update it inside `DATABASE_URL` too).

> **Ordering matters:** the `applicant_data.json` file must be in `~/app/data/`
> **before** `up -d`, or the worker's initial load finds nothing to seed.

---

## 3. Bring the stack up

```bash
cd ~/app
docker compose -f docker-compose.ec2.yml --env-file .env up -d
docker compose -f docker-compose.ec2.yml ps
```

All four services should reach `running` / `healthy`. Capture `ec2-compose-ps.png`.

Images pull automatically. To pin/refresh explicitly:

```bash
docker pull rgogerty/module_6:web-v1
docker pull rgogerty/module_6:worker-v1
```

---

## 4. Verify the live deployment

```bash
# From the instance (or your laptop, since 8080 is open to your IP):
curl -s http://<EC2_PUBLIC_IPV4>:8080/healthz     # -> ok
```

In a browser, open **`http://<EC2_PUBLIC_IPV4>:8080`** and capture `ec2-app.png`
(with the EC2 IP visible in the address bar).

**Prove the background/worker path works in the cloud:** click **Pull Data** in the
UI. The web service publishes a job to RabbitMQ, the **worker** consumes it, runs
the incremental ETL (`SCRAPE_BATCH` new records), and the analytics refresh. Watch it:

```bash
docker compose -f docker-compose.ec2.yml logs -f worker
```

New rows appear and the on-page metrics update — confirming web → RabbitMQ →
worker → Postgres works end to end on EC2.

### Optional: RabbitMQ management UI (via SSH tunnel, never public)

```bash
# From your laptop:
ssh -i key.pem -L 15672:localhost:15672 ubuntu@<EC2_PUBLIC_IPV4>
# then browse http://localhost:15672  (guest / guest)
```

---

## 5. Stop when done (do NOT terminate — reused in Module 8)

```bash
docker compose -f docker-compose.ec2.yml down     # stop containers (pgdata volume persists)
```

Then **Stop** (not Terminate) the instance in the EC2 console so it stops billing
while preserving the machine and its EBS volume.

---

## Troubleshooting

- **`permission denied` on the docker socket** — the group change did not apply.
  Run `newgrp docker` (or log out/in) and retry.
- **`worker` restart-looping / "no data to seed"** — `applicant_data.json` is not in
  `~/app/data/`. Re-run the scp for the data file, then
  `docker compose -f docker-compose.ec2.yml up -d`.
- **`web` unhealthy** — it waits on `db`/`rabbitmq` health. Check
  `docker compose -f docker-compose.ec2.yml ps`; give Postgres a few seconds on
  first boot while `init.sql` runs.
- **Can't reach `:8080` in the browser** — confirm the security group allows 8080
  from *your current* IP (it changes on network switches).
- **`image not found` when pulling** — the images may not be published yet. Build
  and push them from `module_6/` first:

  ```bash
  # from module_6/  (docker login as the rgogerty Docker Hub account)
  docker build -t rgogerty/module_6:web-v1    -f src/web/Dockerfile    src
  docker build -t rgogerty/module_6:worker-v1 -f src/worker/Dockerfile src
  docker push rgogerty/module_6:web-v1
  docker push rgogerty/module_6:worker-v1
  ```
