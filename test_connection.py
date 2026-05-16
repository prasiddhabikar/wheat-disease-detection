import mysql.connector

try:
    # Use the password you set in MySQL Workbench earlier
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_password_here", 
        database="pharmacy_db"
    )

    if mydb.is_connected():
        print("Success! Python is talking to your MySQL database.")
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM medicines")
        
        # This will print the 'Paracetamol' entry we added earlier
        for row in cursor.fetchall():
            print(row)

except Exception as e:
    print(f"Connection failed: {e}")

finally:
    if 'mydb' in locals() and mydb.is_connected():
        mydb.close()