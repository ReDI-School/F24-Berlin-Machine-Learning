CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT,
    amount DECIMAL(10, 2),
    order_date DATE
);

-- Add an index on the 'user_id' column for faster queries involving amount
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Add an index on the 'amount' column for faster queries involving amount
CREATE INDEX idx_orders_amount ON orders(amount);

-- Add an index on the 'order_date' column for faster date range queries
CREATE INDEX idx_orders_order_date ON orders(order_date);
