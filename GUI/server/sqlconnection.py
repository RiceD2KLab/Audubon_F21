import mysql.connector
import csv

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="jose_Mata010913",
  database = "mydatabase"
)

'''
Makes a new table in the database to store data on the birds
'''
def createNewTable():
    newTable = ('''CREATE TABLE `mydatabase`.`birds` (
    `idbirds` INT NOT NULL AUTO_INCREMENT,
    `BirdID` VARCHAR(45) NULL,
    `xCoor` FLOAT NULL,
    `yCoor` FLOAT NULL,
    `imgHeight` FLOAT NULL,
    `imgWidth` FLOAT NULL,
    PRIMARY KEY (`idbirds`))''')
    mycursor.execute(newTable)
    return

'''
Adds in a new bird to the database
'''
def addRow(x1, y1, height, width):
    rowData = f'INSERT INTO `mydatabase`.`birds` (xCoor, yCoor, imgHeight, imgWidth) VALUES ({x1}, {y1}, {height}, {width})'
    print(rowData)
    mycursor.execute(rowData)
    mydb.commit()
    return

'''
Updates the bird species for a specific row in the database
'''
def updateRow(birdNum, birdID):
    rowData = f'UPDATE birds SET BirdID = "{birdID}" WHERE idbirds = {birdNum}'
    mycursor.execute(rowData)
    mydb.commit()
    return

'''
Collects all the data from the database and copies it to a csv file to send to the frontend
'''
def getCSV():
    sql = 'SELECT * from `mydatabase`.`birds`'
    mycursor.execute(sql)
    columnNames = list()
    for i in mycursor.description:
        columnNames.append(i[0])
    rows = mycursor.fetchall()

    result = list()
    result.append(columnNames)
    for row in rows:
        result.append(row)

    fp = open(r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\upload\data.csv', 'w')
    myFile = csv.writer(fp)
    myFile.writerows(result)
    fp.close()

