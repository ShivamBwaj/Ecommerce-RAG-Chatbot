CREATE SCHEMA IF NOT EXISTS shopping_carts;

CREATE TABLE shopping_carts.shopping_cart_items (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    shopping_cart_id VARCHAR(255) NOT NULL DEFAULT 'main',
    product_id VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2),
    quantity INTEGER NOT NULL DEFAULT 1,
    currency VARCHAR(3) DEFAULT 'USD',
    product_image_url VARCHAR(1000),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT positive_price CHECK (price >= 0),
    CONSTRAINT positive_quantity CHECK (quantity > 0),
    CONSTRAINT unique_user_cart_product UNIQUE (user_id, shopping_cart_id, product_id)
);

-- Index for faster queries by user and cart
CREATE INDEX idx_shopping_cart_user_cart ON shopping_carts.shopping_cart_items(user_id, shopping_cart_id);

-- Index for faster queries by user
CREATE INDEX idx_shopping_cart_user_id ON shopping_carts.shopping_cart_items(user_id);

-- Index for faster queries by product
CREATE INDEX idx_shopping_cart_product_id ON shopping_carts.shopping_cart_items(product_id);

-- Trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_shopping_cart_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER shopping_cart_update_timestamp
    BEFORE UPDATE ON shopping_carts.shopping_cart_items
    FOR EACH ROW
    EXECUTE FUNCTION update_shopping_cart_timestamp();