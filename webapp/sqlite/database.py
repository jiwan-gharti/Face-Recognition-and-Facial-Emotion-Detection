import sqlite3



conn = sqlite3.connect('face.db',check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS face_recognition(ID INTEGER PRIMARY KEY AUTOINCREMENT,NAME TEXT(50),AGE TEXT(50),GENDER TEXT(50),Image TEXT(50) NULL);")
conn.commit()

def add_data(*args, **kwargs):
    cur.execute("INSERT INTO face_recognition(NAME,AGE,GENDER) VALUES(?,?,?)",(args[0],args[1],args[2]))
    conn.commit()
    return True

def view_all_data():
    cur.execute("SELECT * from face_recognition")
    return cur.fetchall()
    

def get_single_data(id):
    cur.execute("SELECT * FROM face_recognition where id={}".format(id))
    data =  cur.fetchone()
    return data
    

def edit_data(*args, **kwargs):
    cur.execute('UPDATE face_recognition SET name=?, age=?, gender=?,Image=? WHERE id=?',(args[1],args[2],args[3],args[4],args[0]))
    conn.commit()


def delete(id):
    cur.execute(f"DELETE FROM face_recognition WHERE id={id}")
    conn.commit()
    