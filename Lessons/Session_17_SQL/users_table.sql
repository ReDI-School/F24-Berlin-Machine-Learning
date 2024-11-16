CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    age INT,
    join_date DATE
);

-- Add an index on the 'age' column for faster queries filtering by age
CREATE INDEX idx_users_age ON users(age);

-- Add an index on the 'join_date' column for faster range queries
CREATE INDEX idx_users_join_date ON users(join_date);
