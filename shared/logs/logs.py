import datetime

class Logger:
    # Codes ANSI pour le style
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    def __init__(self, show_time=True):
        self.show_time = show_time

    def _get_prefix(self, label, color):
        """Génère le préfixe avec horodatage et couleur."""
        time_str = f"{datetime.datetime.now().strftime('%H:%M:%S')} " if self.show_time else ""
        return f"{self.BOLD}{time_str}{color}[{label}]{self.RESET}"

    def success(self, message: str):
        print(f"{self._get_prefix('SUCCESS', self.GREEN)} {message}")

    def error(self, message: str):
        print(f"{self._get_prefix('ERROR', self.RED)} {message}")

    def warning(self, message: str):
        print(f"{self._get_prefix('WARNING', self.YELLOW)} {message}")

    def info(self, message: str):
        print(f"{self._get_prefix('INFO', self.BLUE)} {message}")

# --- Test de la classe ---
"""
log = Logger()

log.info("Initialisation du système...")
log.success("Base de données connectée.")
log.warning("Utilisation CPU élevée.")
log.error("Échec de la sauvegarde des données.")
"""