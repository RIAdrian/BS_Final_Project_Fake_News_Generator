import streamlit as st
from datetime import datetime
import re
import sqlite3
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import random
import csv
import os

# Funcții de validare
def validate_email(email):
    return "@" in email

def validate_username(username):
    return re.match("^[A-Za-z][A-Za-z0-9_]*$", username) is not None

def validate_password(password):
    return len(password) >= 8

# Funcție pentru a verifica dacă username-ul există deja
def username_exists(username, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM users WHERE username = ?', (username,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

# Funcție pentru a verifica autentificarea utilizatorului
def authenticate_user(username, password, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT id, password, email FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result is None:
        return False, "Username-ul nu există.", None, None
    elif result[1] != password:
        return False, "Parola este incorectă.", None, None
    return True, None, result[0], result[2]

# Funcție pentru a verifica datele pentru resetarea parolei
def validate_reset_password(username, email, current_password, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT email, password FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result is None:
        return False, "Username-ul nu există."
    elif result[0] != email:
        return False, "Emailul nu corespunde cu username-ul."
    elif result[1] != current_password:
        return False, "Parola curentă este incorectă."
    return True, None

# Funcție pentru a reseta parola
def reset_password(username, new_password, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET password = ? WHERE username = ?', (new_password, username))
    conn.commit()
    conn.close()

# Funcție pentru a crea un nou cont
def create_account(email, username, password, date_of_birth, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (email, username, password, date_of_birth) VALUES (?, ?, ?, ?)', 
                       (email, username, password, date_of_birth))
        conn.commit()
        user_id = cursor.lastrowid
        return True, user_id
    except sqlite3.IntegrityError:
        return False, None
    finally:
        conn.close()

# Inițializare baza de date
def init_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            date_of_birth DATE NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS continut_antrenare (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

# Apel inițializare baze de date
init_db('users_admin.db')
init_db('users_client.db')

# Încărcare model GPT-2
@st.cache_resource
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model("model_salvat2")
model_train, tokenizer_train = load_model("model_salvat")

# Setare token de padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer_train.pad_token = tokenizer_train.eos_token

# Funcție pentru generarea textului
def generate_text(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=300, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Funcție pentru numărarea textelor din baza de date
def count_texts_in_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM continut_antrenare')
    count = cursor.fetchone()[0]
    conn.close()
    return count

# Funcție pentru obținerea textelor din baza de date
def get_texts_from_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT text FROM continut_antrenare')
    texts = [row[0] for row in cursor.fetchall()]
    conn.close()
    return texts

# Funcție pentru antrenarea modelului
def train_model(model, tokenizer, texts):
    # Configurare optimizator și program de învățare
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(texts) * 3  # presupunem 3 epoci
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    model.train()
    for epoch in range(3):
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

# Citirea fișierului Excel
def load_excel(file_path):
    df = pd.read_excel(file_path)
    df['Tag'] = df['Tag'].str.upper()  # Normalizează valorile Tag la majuscule
    return df

# Selectarea random a 10 texte din DataFrame
def get_random_texts(df, num_texts=10):
    return df.sample(n=num_texts).reset_index(drop=True)

# Verificarea scorului
def calculate_score(selected_answers, correct_answers):
    score = 0
    for selected, correct in zip(selected_answers, correct_answers):
        if selected == correct:
            score += 1
    return score

# Funcție pentru generarea raportului
def generate_report(questions, user_answers, correct_answers, username, email, file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Întrebare", "Răspunsul utilizatorului", "Răspunsul corect"])
        for question, user_answer, correct_answer in zip(questions, user_answers, correct_answers):
            writer.writerow([question, user_answer, correct_answer])
        writer.writerow([])  # Linie goală pentru separare
        writer.writerow(["Username", username])
        writer.writerow(["Email", email])
        writer.writerow(["Data finalizării", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# Stiluri CSS
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #2E2E2E;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 2rem;
    }
    .sidebar .sidebar-content .block-container .markdown-text-container {
        text-align: left;
    }
    .sidebar .sidebar-content .block-container h2 {
        color: white;
    }
    .sidebar .sidebar-content .block-container p {
        color: white;
    }
    .sidebar .sidebar-content .block-container button {
        background-color: #4A4A4A;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 5px;
        text-align: left;
    }
    .sidebar .sidebar-content .block-container button:hover {
        background-color: #6A6A6A;
    }
    .sidebar .sidebar-content .block-container .selected {
        background-color: #C0C0C0 !important;
        color: black !important;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Titlul paginii principale
st.markdown("<h1 style='text-align: center; color: white;'>Generator de Fake News</h1>", unsafe_allow_html=True)

# Bara laterală de navigare
def show_sidebar():
    with st.sidebar:
        st.header("Navigare")
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False

        if st.session_state.authenticated:
            if st.session_state.get('user_type') == 'admin':
                st.session_state.page = st.session_state.get('page', 'generare')
                if st.button("Generare"):
                    st.session_state.page = 'generare'
                    st.rerun()
                if st.button("Antrenare"):
                    st.session_state.page = 'antrenare'
                    st.rerun()
                if st.button("Log out"):
                    st.session_state.page = 'logout_confirm'
                    st.rerun()
            elif st.session_state.get('user_type') == 'client':
                st.session_state.page = st.session_state.get('page', 'pagina_principala')
                if st.button("Pagina Principala"):
                    st.session_state.page = 'pagina_principala'
                    st.rerun()
                if st.button("Log out"):
                    st.session_state.page = 'logout_confirm'
                    st.rerun()
        else:
            if 'page' not in st.session_state:
                st.session_state.page = 'login_admin'
            
            login_button_admin = st.button("Logare pentru Admin", key="login_button_admin", help="Login to your admin account")
            create_account_button_admin = st.button("Înregistrare pentru Admin", key="create_account_button_admin", help="Create a new admin account")
            login_button_client = st.button("Logare pentru Client", key="login_button_client", help="Login to your client account")
            create_account_button_client = st.button("Înregistrare pentru Client", key="create_account_button_client", help="Create a new client account")
            forgot_password_button = st.button("Recuperare Parola?", key="forgot_password_button", help="Recover your password")
            reset_password_button = st.button("Resetare Parola", key="reset_password_button", help="Reset your password")

            if login_button_admin:
                st.session_state.page = 'login_admin'
                st.rerun()
            elif create_account_button_admin:
                st.session_state.page = 'create_account_admin'
                st.rerun()
            elif login_button_client:
                st.session_state.page = 'login_client'
                st.rerun()
            elif create_account_button_client:
                st.session_state.page = 'create_account_client'
                st.rerun()
            elif forgot_password_button:
                st.session_state.page = 'forgot_password'
                st.rerun()
            elif reset_password_button:
                st.session_state.page = 'reset_password'
                st.rerun()

# Afișare bara laterală doar dacă nu suntem în modul aplicație
show_sidebar()

# Conținut principal
if st.session_state.get('page') == 'login_admin':
    st.header("Logare pentru Admin")
    username = st.text_input("Username", placeholder="Username unic")
    password = st.text_input("Password", type="password", placeholder="Parola dumneavoastră dorită")
    if st.button("Login"):
        is_authenticated, error_message, user_id, email = authenticate_user(username, password, 'users_admin.db')
        if is_authenticated:
            st.session_state.authenticated = True
            st.session_state.page = 'generare'
            st.session_state.user_id = user_id
            st.session_state.user_type = 'admin'
            st.session_state.username = username  # Store username
            st.session_state.email = email  # Store email
            st.rerun()
        else:
            st.error(error_message)

elif st.session_state.get('page') == 'create_account_admin':
    st.header("Înregistrare pentru Admin")
    data_nastere = st.date_input("Data de naștere", min_value=datetime(1914, 1, 1), max_value=datetime.now())
    
    email = st.text_input("Email", placeholder="Adresa dumneavoastră de email")
    if email and not validate_email(email):
        st.error("Emailul trebuie să conțină caracterul '@'.")
    
    username = st.text_input("Username", placeholder="Nu trebuie să înceapă cu o cifră sau caracter special și să nu conțină spații")
    if username and not validate_username(username):
        st.error("Username-ul nu trebuie să înceapă cu o cifră sau caracter special și să nu conțină spații.")
    
    parola = st.text_input("Parolă", type="password", placeholder="Parola trebuie să conțină minim 8 caractere")
    if parola and not validate_password(parola):
        st.error("Parola trebuie să conțină minim 8 caractere.")
    
    confirm_parola = st.text_input("Confirmă Parolă", type="password", placeholder="Re-introduceți parola")
    if confirm_parola and parola != confirm_parola:
        st.error("Parolele nu se potrivesc.")
    
    if st.button("Creare cont"):
        if not validate_email(email):
            st.error("Email invalid.")
        elif not validate_username(username):
            st.error("Username invalid.")
        elif username_exists(username, 'users_admin.db'):
            st.error("Username-ul este deja folosit. Alegeți un alt username.")
        elif not validate_password(parola):
            st.error("Parola trebuie să conțină minim 8 caractere.")
        elif parola != confirm_parola:
            st.error("Parolele nu se potrivesc.")
        else:
            success, user_id = create_account(email, username, parola, data_nastere, 'users_admin.db')
            if success:
                st.session_state.authenticated = True
                st.session_state.page = 'generare'
                st.session_state.user_id = user_id
                st.session_state.user_type = 'admin'
                st.session_state.username = username  # Store username
                st.session_state.email = email  # Store email
                st.success("Contul a fost creat cu succes!")
                st.rerun()
            else:
                st.error("A apărut o eroare. Încercați din nou.")

elif st.session_state.get('page') == 'login_client':
    st.header("Logare pentru Client")
    username = st.text_input("Username", placeholder="Username unic")
    password = st.text_input("Password", type="password", placeholder="Parola dumneavoastră dorită")
    if st.button("Login"):
        is_authenticated, error_message, user_id, email = authenticate_user(username, password, 'users_client.db')
        if is_authenticated:
            st.session_state.authenticated = True
            st.session_state.page = 'pagina_principala'
            st.session_state.user_id = user_id
            st.session_state.user_type = 'client'
            st.session_state.username = username  # Store username
            st.session_state.email = email  # Store email
            st.rerun()
        else:
            st.error(error_message)

elif st.session_state.get('page') == 'create_account_client':
    st.header("Înregistrare pentru Client")
    data_nastere = st.date_input("Data de naștere", min_value=datetime(1914, 1, 1), max_value=datetime.now())
    
    email = st.text_input("Email", placeholder="Adresa dumneavoastră de email")
    if email and not validate_email(email):
        st.error("Emailul trebuie să conțină caracterul '@'.")
    
    username = st.text_input("Username", placeholder="Nu trebuie să înceapă cu o cifră sau caracter special și să nu conțină spații")
    if username and not validate_username(username):
        st.error("Username-ul nu trebuie să înceapă cu o cifră sau caracter special și să nu conțină spații.")
    
    parola = st.text_input("Parolă", type="password", placeholder="Parola trebuie să conțină minim 8 caractere")
    if parola and not validate_password(parola):
        st.error("Parola trebuie să conțină minim 8 caractere.")
    
    confirm_parola = st.text_input("Confirmă Parolă", type="password", placeholder="Re-introduceți parola")
    if confirm_parola and parola != confirm_parola:
        st.error("Parolele nu se potrivesc.")
    
    if st.button("Creare cont"):
        if not validate_email(email):
            st.error("Email invalid.")
        elif not validate_username(username):
            st.error("Username invalid.")
        elif username_exists(username, 'users_client.db'):
            st.error("Username-ul este deja folosit. Alegeți un alt username.")
        elif not validate_password(parola):
            st.error("Parola trebuie să conțină minim 8 caractere.")
        elif parola != confirm_parola:
            st.error("Parolele nu se potrivesc.")
        else:
            success, user_id = create_account(email, username, parola, data_nastere, 'users_client.db')
            if success:
                st.session_state.authenticated = True
                st.session_state.page = 'pagina_principala'
                st.session_state.user_id = user_id
                st.session_state.user_type = 'client'
                st.session_state.username = username  # Store username
                st.session_state.email = email  # Store email
                st.success("Contul a fost creat cu succes!")
                st.rerun()
            else:
                st.error("A apărut o eroare. Încercați din nou.")

elif st.session_state.get('page') == 'pagina_principala':
    st.header("Test de recunoaștere a știrilor false")

    # Încărcăm fișierul Excel și selectăm 10 texte random
    if 'selected_texts' not in st.session_state:
        df = load_excel('Baza_teste.xlsx')
        st.session_state.selected_texts = get_random_texts(df)
        st.session_state.selected_answers = [None] * 10

    selected_texts = st.session_state.selected_texts

    for i in range(10):
        text = selected_texts.loc[i, 'Text']
        st.write(f"Text {i+1}: {text}")

        adevarat_checked = st.checkbox("ADEVARAT", key=f"adevarat_{i}")
        fals_checked = st.checkbox("FALS", key=f"fals_{i}")

        if adevarat_checked and not fals_checked:
            st.session_state.selected_answers[i] = 'ADEVARAT'
        elif fals_checked and not adevarat_checked:
            st.session_state.selected_answers[i] = 'FALS'
        else:
            st.session_state.selected_answers[i] = None

    if st.button("Verificare scor"):
        correct_answers = selected_texts['Tag'].tolist()
        st.write(f"Răspunsurile corecte sunt: {correct_answers}")  # Debug: Afișează răspunsurile corecte
        st.write(f"Răspunsurile selectate sunt: {st.session_state.selected_answers}")  # Debug: Afișează răspunsurile selectate
        score = calculate_score(st.session_state.selected_answers, correct_answers)
        st.success(f"Scorul dumneavoastră este: {score}/10")
        
        # Generăm raportul și îl salvăm într-un fișier CSV
        report_path = "raport_test.csv"
        questions = selected_texts['Text'].tolist()
        generate_report(questions, st.session_state.selected_answers, correct_answers, st.session_state.username, st.session_state.email, report_path)
        
        # Afișăm butonul de descărcare
        with open(report_path, 'rb') as file:
            btn = st.download_button(
                label="Descarcă Raport Test",
                data=file,
                file_name=report_path,
                mime='text/csv'
            )

elif st.session_state.get('page') == 'forgot_password':
    st.header("Recuperați parola")
    email = st.text_input("Email", placeholder="Adresa dumneavoastră de email")
    if st.button("Trimite"):
        st.success("Un email de recuperare a fost trimis!")

elif st.session_state.get('page') == 'reset_password':
    st.header("Resetați parola")
    username = st.text_input("Username", placeholder="Username-ul dumneavoastră")
    email = st.text_input("Email", placeholder="Adresa dumneavoastră de email")
    current_password = st.text_input("Parola curentă", type="password", placeholder="Introduceți parola curentă")
    new_password = st.text_input("Noua parolă", type="password", placeholder="Introduceți noua parolă")
    confirm_password = st.text_input("Confirmă noua parolă", type="password", placeholder="Confirmă noua parolă")
    if st.button("Resetează parola"):
        if new_password != confirm_password:
            st.error("Parolele nu se potrivesc. Încearcă din nou.")
        else:
            is_valid, error_message = validate_reset_password(username, email, current_password, 'users_admin.db')  # Sau 'users_client.db' în funcție de utilizator
            if is_valid:
                reset_password(username, new_password, 'users_admin.db')  # Sau 'users_client.db' în funcție de utilizator
                st.success("Parola a fost resetată cu succes!")
            else:
                st.error(error_message)

elif st.session_state.get('page') == 'generare':
    st.header("Generare text")
    
    if 'num_prompts' not in st.session_state:
        st.session_state.num_prompts = 0
    
    if 'prompts' not in st.session_state:
        st.session_state.prompts = []

    if 'generated_texts' not in st.session_state:
        st.session_state.generated_texts = []

    num_prompts = st.text_input("Câte generări de text dorești?", "")
    if num_prompts.isdigit():
        st.session_state.num_prompts = int(num_prompts)
    else:
        st.error("Te rugăm să introduci un număr valid.")
    
    prompts = st.session_state.prompts

    if st.session_state.num_prompts > 0:
        for i in range(st.session_state.num_prompts):
            if i < len(prompts):
                prompt = st.text_input(f"Prompt {i+1}", value=prompts[i], key=f"prompt_{i}")
            else:
                prompt = st.text_input(f"Prompt {i+1}", key=f"prompt_{i}")
                prompts.append(prompt)
            prompts[i] = prompt

        if st.button("Generare Text"):
            st.session_state.generated_texts = []
            for prompt in prompts:
                if prompt:
                    with st.spinner(f"Generare text pentru: {prompt}..."):
                        generated_text = generate_text(prompt, model, tokenizer)
                        st.session_state.generated_texts.append((prompt, generated_text))
            st.session_state.show_add_text = True
            st.rerun()

    if st.session_state.get('show_add_text', False):
        for prompt, generated_text in st.session_state.generated_texts:
            st.write(f"**Prompt:** {prompt}")
            st.write(f"**Text generat:** {generated_text}")

        if 'add_text' not in st.session_state:
            st.session_state.add_text = ""

        st.markdown("### Doriți să adăugați textul în baza de date pentru antrenamente viitoare?")
        st.session_state.add_text = st.text_area("Introduceți textul pentru a-l adăuga în baza de date", value=st.session_state.add_text)
        if st.button("Adaugă în DB"):
            if st.session_state.add_text:
                conn = sqlite3.connect('users_admin.db')  
                cursor = conn.cursor()
                user_id = st.session_state.user_id
                cursor.execute('INSERT INTO continut_antrenare (user_id, text) VALUES (?, ?)', (user_id, st.session_state.add_text))
                conn.commit()
                conn.close()
                st.success("Textul a fost adăugat în baza de date.")
                st.session_state.show_add_text = False
                st.session_state.add_text = ""
                # st.rerun()
            else:
                st.error("Textul nu poate fi gol. Introduceți un text valid.")

elif st.session_state.get('page') == 'antrenare':
    st.header("Antrenare model")
    
    num_texts = count_texts_in_db('users_admin.db')  
    st.write(f"În momentul de față aveți {num_texts} texte în baza de antrenare.")
    
    uploaded_file = st.file_uploader("Alegeți un fișier de antrenare", type=["txt"])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        conn = sqlite3.connect('users_admin.db')  
        cursor = conn.cursor()
        user_id = st.session_state.user_id
        cursor.execute('INSERT INTO continut_antrenare (user_id, text) VALUES (?, ?)', (user_id, file_content))
        conn.commit()
        conn.close()
        st.success("Fișierul a fost adăugat în baza de date.")
    
    if st.button("Antrenează modelul"):
        texts = get_texts_from_db('users_admin.db')  
        with st.spinner("Modelul se antrenează..."):
            train_model(model_train, tokenizer_train, texts)
        st.success("Modelul a fost antrenat cu succes.")

elif st.session_state.get('page') == 'logout_confirm':
    st.header("Sunteți sigur că doriți să ieșiți din cont?")
    if st.button("Da, sunt sigur"):
        st.session_state.authenticated = False
        st.session_state.page = 'login_admin'  
        st.rerun()
