import mysql.connector
from mysql.connector import Error
from .models import Face, Entry
from config.database import DB_CONFIG

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def connect(self):
        try:
            self.conn = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            
    def setup_tables(self):
        # [Previous table creation code here]
        pass