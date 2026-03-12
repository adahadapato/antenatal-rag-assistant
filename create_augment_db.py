# create_augment_db.py This script creates the SQLite database for storing antenatal condition information.
# Import sqlite3 so we can create a SQLite database.
import sqlite3
import Initialize

# Store the database file path in a variable.
#doc_path = Initialize.doc_path
db_path = Initialize.augmented_db_path


# Connect to the SQLite database.
# If the file does not exist, SQLite creates it automatically.
conn = sqlite3.connect(db_path)

# Create a cursor object to execute SQL commands.
cursor = conn.cursor()

# Create the main conditions table.
cursor.execute("""
CREATE TABLE IF NOT EXISTS conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    policy_name TEXT,
    policy_version TEXT,
    source_page INTEGER
)
""")

# Create the recommendations table.
cursor.execute("""
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (condition_id) REFERENCES conditions(id)
)
""")

# Create the tests table.
cursor.execute("""
CREATE TABLE IF NOT EXISTS tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (condition_id) REFERENCES conditions(id)
)
""")

# Create the ultrasound table.
cursor.execute("""
CREATE TABLE IF NOT EXISTS ultrasound (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (condition_id) REFERENCES conditions(id)
)
""")

# Create the reasons table.
cursor.execute("""
CREATE TABLE IF NOT EXISTS reasons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (condition_id) REFERENCES conditions(id)
)
""")

# Save all changes.
conn.commit()

# Close the connection.
conn.close()

# Print success message.
print("argumentation.db created successfully.")