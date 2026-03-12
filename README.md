# detection_champion_lol

Architecture cible du projet : 
lol-realtime-detector/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   ├── protocol.md
│   └── deployment.md
├── shared/
│   ├── protocol/
│   │   ├── detection_schema.json
│   │   └── messages.proto
│   └── utils/
├── pc-client/
│   ├── src/
│   │   ├── capture/
│   │   ├── stream/
│   │   ├── overlay/
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── jetson-server/
│   ├── src/
│   │   ├── ingest/
│   │   ├── infer/
│   │   ├── postprocess/
│   │   ├── transport/
│   │   └── main.py
│   ├── models/
│   ├── benchmarks/
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   ├── run_pc.sh
│   ├── run_jetson.sh
│   └── export_tensorrt.sh
└── .github/
    └── workflows/


Serveur d'inférence : 
Flow du serveur d'inférence

Démarrage du serveur
1. Créer LatestFrameBuffer
2. Charger YoloDetector et faire le warmup
3. Démarrer le serveur réseau
4. Accepter un client
5. Lancer le thread de réception
6. Lancer le thread d’inférence

Thread de réception
1. Tant que le serveur tourne et que le client est connecté :
   a. recevoir une frame
   b. créer un FramePacket(frame_id, timestamp, frame)
   c. déposer/remplacer dans LatestFrameBuffer

Thread d’inférence
1. Tant que le serveur tourne :
   a. attendre une frame disponible
   b. récupérer le dernier FramePacket
   c. exécuter l’inférence
   d. construire DetectionMessage
   e. envoyer le message au client

Arrêt
1. Détecter fermeture client / erreur / arrêt demandé
2. Passer running à False
3. Réveiller les threads bloqués si besoin
4. Fermer sockets et joindre les threads
