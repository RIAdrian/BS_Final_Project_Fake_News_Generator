import sqlite3

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Eliminăm tabelele existente, dacă există
    cursor.execute('''
        DROP TABLE IF EXISTS users
    ''')
    cursor.execute('''
        DROP TABLE IF EXISTS continut_antrenare
    ''')
    # Creăm tabela users cu constrângerea de unicitate
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            date_of_birth DATE NOT NULL
        )
    ''')
    # Creăm tabela continut_antrenare
    cursor.execute('''
        CREATE TABLE continut_antrenare (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
