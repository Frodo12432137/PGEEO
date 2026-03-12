import pyodbc

servers = ["MISDWHPRD.GKPGE.PL", "MISDWPPRD.GKPGE.PL"]
databases = ["PGESA_MarketAnalytics", "PGEEO_DDS"]

print("=== DIAGNOSTYKA POŁĄCZENIA SQL SERVER ===\n")

for server in servers:
    print(f"Sprawdzanie serwera: {server}...")
    for db in databases:
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"Server={server};"
            f"DATABASE={db};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )
        try:
            print(f"  - Próba połączenia z bazą: {db}...", end=" ", flush=True)
            conn = pyodbc.connect(conn_str, timeout=5)
            print("OK!")
            conn.close()
        except pyodbc.Error as e:
            print(f"BŁĄD")
            print(f"    Opis: {str(e)[:200]}...")

print("\n=== KONIEC DIAGNOSTYKI ===")
