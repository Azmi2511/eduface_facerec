const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',     
    password: '',      
    database: 'eduface' 
});

db.connect(err => {
    if (err) console.error('Koneksi Database Gagal: ', err);
    else console.log('Terhubung ke MySQL');
});

app.get('/api/students', (req, res) => {
    const sql = "SELECT nisn, full_name FROM students ORDER BY full_name ASC";
    db.query(sql, (err, results) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json(results);
    });
});

app.get('/api/registered-students', (req, res) => {
    const sql = `
        SELECT nisn, full_name, face_registered_at 
        FROM students 
        WHERE is_face_registered = 1 
        ORDER BY face_registered_at DESC
    `;
    
    db.query(sql, (err, results) => {
        if (err) {
            console.error("Database Error:", err);
            return res.status(500).json({ error: err.message });
        }
        res.json(results);
    });
});

app.listen(3000, () => {
    console.log('Server berjalan di http://localhost:3000');
});