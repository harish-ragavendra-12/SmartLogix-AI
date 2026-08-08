-- ==========================================================
-- SMARTLOGIX AI
-- FINAL DATABASE SCHEMA
-- PostgreSQL
-- ==========================================================

-- ==========================================================
-- 1. CUSTOMERS
-- ==========================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id        VARCHAR(50) PRIMARY KEY,
    customer_name      VARCHAR(150),
    email              VARCHAR(150),
    phone              VARCHAR(30),
    city               VARCHAR(100),
    state              VARCHAR(100),
    pincode             VARCHAR(20),
    signup_date         DATE,
    customer_segment    VARCHAR(50),
    is_prime_member     BOOLEAN,
    lifetime_orders     INTEGER,
    avg_rating_given    NUMERIC(3, 2)
);


-- ==========================================================
-- 2. PRODUCTS
-- ==========================================================

CREATE TABLE IF NOT EXISTS products (
    product_id              VARCHAR(50) PRIMARY KEY,
    product_name            VARCHAR(255),
    category                VARCHAR(100),
    sub_category            VARCHAR(100),

    is_fragile              BOOLEAN,
    is_hazmat               BOOLEAN,
    requires_cold_chain     BOOLEAN,

    stock_qty               INTEGER,
    avg_rating              NUMERIC(3, 2),

    tags                    TEXT[],

    launch_date             DATE,

    weight_kg               NUMERIC(10, 3),
    battery_included        BOOLEAN,

    price_currency          VARCHAR(10),
    price_amount            NUMERIC(12, 2),

    dimension_length_cm     NUMERIC(10, 2),
    dimension_width_cm      NUMERIC(10, 2),
    dimension_height_cm     NUMERIC(10, 2)
);


-- ==========================================================
-- 3. FLEET VEHICLES
-- ==========================================================

CREATE TABLE IF NOT EXISTS fleet_vehicles (
    vehicle_id            VARCHAR(50) PRIMARY KEY,
    vehicle_type          VARCHAR(50),
    model_name            VARCHAR(100),

    capacity_kg            NUMERIC(10, 2),
    max_range_km           NUMERIC(10, 2),
    avg_speed_kmph         NUMERIC(10, 2),

    hub_code               VARCHAR(50),

    purchase_date          DATE,
    odometer_km            NUMERIC(12, 2),

    fleet_status           VARCHAR(50),
    driver_id              VARCHAR(50),

    battery_capacity_wh    NUMERIC(12, 2),

    last_service_date      DATE,
    insurance_expiry       DATE,

    ownership              VARCHAR(50)
);


-- ==========================================================
-- 4. ORDERS
-- ==========================================================

CREATE TABLE IF NOT EXISTS orders (
    order_id                    VARCHAR(50) PRIMARY KEY,

    customer_id                 VARCHAR(50),
    product_id                  VARCHAR(50),

    quantity                    INTEGER,

    order_date                  TIMESTAMP,

    origin_hub                  VARCHAR(100),
    origin_city                 VARCHAR(100),

    destination_city            VARCHAR(100),
    destination_state           VARCHAR(100),
    destination_pincode         VARCHAR(20),

    destination_lat             NUMERIC(10, 7),
    destination_lon             NUMERIC(10, 7),

    distance_km                 NUMERIC(10, 2),
    package_weight_kg           NUMERIC(10, 3),

    package_dimensions_cm       VARCHAR(100),

    is_fragile                  BOOLEAN,
    is_hazmat                   BOOLEAN,
    cold_chain_required         BOOLEAN,

    delivery_priority           VARCHAR(50),
    payment_mode                VARCHAR(50),

    order_value_inr             NUMERIC(12, 2),

    weather_condition_at_dest   VARCHAR(100),

    assigned_vehicle_id         VARCHAR(50),

    promised_eta_hours          NUMERIC(10, 2),
    actual_delivery_hours       NUMERIC(10, 2),

    delivery_cost_inr           NUMERIC(12, 2),

    transport_mode              VARCHAR(50),
    order_status                VARCHAR(50),

    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id),

    CONSTRAINT fk_orders_product
        FOREIGN KEY (product_id)
        REFERENCES products(product_id),

    CONSTRAINT fk_orders_vehicle
        FOREIGN KEY (assigned_vehicle_id)
        REFERENCES fleet_vehicles(vehicle_id)
);


-- ==========================================================
-- 5. DELIVERY LOGS
-- ==========================================================

CREATE TABLE IF NOT EXISTS delivery_logs (
    log_id              VARCHAR(50) PRIMARY KEY,

    order_id            VARCHAR(50) NOT NULL,

    event_seq           INTEGER,

    event_type          VARCHAR(100),

    event_timestamp     TIMESTAMP,

    hub_code             VARCHAR(50),

    vehicle_id           VARCHAR(50),

    scanned_by_emp       VARCHAR(50),

    location_city        VARCHAR(100),

    remarks              TEXT,

    exception_code       VARCHAR(50),

    CONSTRAINT fk_delivery_logs_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    CONSTRAINT fk_delivery_logs_vehicle
        FOREIGN KEY (vehicle_id)
        REFERENCES fleet_vehicles(vehicle_id)
);


-- ==========================================================
-- 6. DRONE TELEMETRY
-- ==========================================================

CREATE TABLE IF NOT EXISTS drone_telemetry (
    flight_id                VARCHAR(50) PRIMARY KEY,

    drone_id                 VARCHAR(50) NOT NULL,

    flight_timestamp         TIMESTAMP,

    flight_duration_min      NUMERIC(10, 2),
    cumulative_flight_hours  NUMERIC(12, 2),

    battery_cycles           INTEGER,
    battery_start_pct        NUMERIC(5, 2),
    battery_end_pct          NUMERIC(5, 2),
    battery_health_pct       NUMERIC(5, 2),

    motor_temp_c             NUMERIC(10, 2),
    vibration_rms            NUMERIC(10, 3),
    payload_kg               NUMERIC(10, 3),

    max_altitude_m           NUMERIC(10, 2),
    wind_speed_kmph          NUMERIC(10, 2),

    gps_signal_quality       VARCHAR(50),

    rotor_rpm_avg            NUMERIC(12, 2),

    error_codes              TEXT,

    route_deviation_m        NUMERIC(10, 2),

    maintenance_required     BOOLEAN,

    CONSTRAINT fk_drone_telemetry_vehicle
        FOREIGN KEY (drone_id)
        REFERENCES fleet_vehicles(vehicle_id)
);


-- ==========================================================
-- 7. MAINTENANCE HISTORY
-- ==========================================================

CREATE TABLE IF NOT EXISTS maintenance_history (
    work_order_id       VARCHAR(50) PRIMARY KEY,

    vehicle_id          VARCHAR(50) NOT NULL,

    service_date        DATE,

    service_type        VARCHAR(100),

    parts_replaced      TEXT,

    labour_hours        NUMERIC(10, 2),

    cost_inr            NUMERIC(12, 2),

    downtime_hours      NUMERIC(10, 2),

    failure_reported    BOOLEAN,

    technician_id       VARCHAR(50),

    CONSTRAINT fk_maintenance_vehicle
        FOREIGN KEY (vehicle_id)
        REFERENCES fleet_vehicles(vehicle_id)
);


-- ==========================================================
-- 8. GPS ROUTES
-- ==========================================================

CREATE TABLE IF NOT EXISTS gps_routes (
    route_id               VARCHAR(50) PRIMARY KEY,

    vehicle_id             VARCHAR(50) NOT NULL,

    vehicle_type           VARCHAR(50),

    route_date             DATE,

    planned_distance_km    NUMERIC(10, 2),

    actual_distance_km     NUMERIC(10, 2),

    planned_duration_min   NUMERIC(10, 2),

    actual_duration_min    NUMERIC(10, 2),

    stops_planned          INTEGER,

    stops_completed        INTEGER,

    fuel_or_energy_used    NUMERIC(12, 2),

    energy_unit            VARCHAR(20),

    driver_id              VARCHAR(50),

    CONSTRAINT fk_gps_routes_vehicle
        FOREIGN KEY (vehicle_id)
        REFERENCES fleet_vehicles(vehicle_id)
);


-- ==========================================================
-- 9. ROUTE ORDERS
-- ==========================================================

CREATE TABLE IF NOT EXISTS route_orders (
    route_id          VARCHAR(50),
    order_id          VARCHAR(50),

    route_order_seq   INTEGER,

    PRIMARY KEY (route_id, order_id),

    CONSTRAINT fk_route_orders_route
        FOREIGN KEY (route_id)
        REFERENCES gps_routes(route_id),

    CONSTRAINT fk_route_orders_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
);


-- ==========================================================
-- 10. ROUTE WAYPOINTS
-- ==========================================================

CREATE TABLE IF NOT EXISTS route_waypoints (
    route_id             VARCHAR(50),

    waypoint_seq         INTEGER,

    latitude             NUMERIC(10, 7),
    longitude            NUMERIC(10, 7),

    waypoint_timestamp   TIMESTAMP,

    speed_kmph           NUMERIC(10, 2),

    PRIMARY KEY (route_id, waypoint_seq),

    CONSTRAINT fk_route_waypoints_route
        FOREIGN KEY (route_id)
        REFERENCES gps_routes(route_id)
);


-- ==========================================================
-- 11. TRAFFIC DATA
-- ==========================================================

CREATE TABLE IF NOT EXISTS traffic_data (
    traffic_id          BIGSERIAL PRIMARY KEY,

    record_date         DATE,

    hour_of_day         INTEGER,

    city                VARCHAR(100),

    corridor            VARCHAR(150),

    congestion_index    NUMERIC(5, 3),

    avg_speed_kmph      NUMERIC(10, 2),

    incident_reported   BOOLEAN,

    road_closure        BOOLEAN
);


-- ==========================================================
-- 12. WEATHER DATA
-- ==========================================================

CREATE TABLE IF NOT EXISTS weather_data (
    weather_id          BIGSERIAL PRIMARY KEY,

    observation_date    DATE,

    city                VARCHAR(100),

    temperature_c       NUMERIC(6, 2),

    humidity_pct        NUMERIC(5, 2),

    precipitation_mm    NUMERIC(10, 2),

    wind_speed_kmph     NUMERIC(10, 2),

    visibility_km       NUMERIC(10, 2),

    condition           VARCHAR(100),

    storm_alert         BOOLEAN
);


-- ==========================================================
-- 13. CUSTOMER REVIEWS
-- ==========================================================

CREATE TABLE IF NOT EXISTS customer_reviews (
    review_id                  VARCHAR(50) PRIMARY KEY,

    order_id                   VARCHAR(50),

    product_id                 VARCHAR(50),

    customer_id                VARCHAR(50),

    rating                     NUMERIC(3, 2),

    review_title               VARCHAR(255),

    review_text                TEXT,

    review_date                TIMESTAMP,

    verified_purchase          BOOLEAN,

    helpful_votes              INTEGER,

    delivery_mode_experienced  VARCHAR(50),

    sentiment_label            VARCHAR(20),

    CONSTRAINT fk_reviews_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    CONSTRAINT fk_reviews_product
        FOREIGN KEY (product_id)
        REFERENCES products(product_id),

    CONSTRAINT fk_reviews_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);


-- ==========================================================
-- INDEXES
-- ==========================================================

-- Customers
CREATE INDEX IF NOT EXISTS idx_customers_city
    ON customers(city);

CREATE INDEX IF NOT EXISTS idx_customers_segment
    ON customers(customer_segment);


-- Products
CREATE INDEX IF NOT EXISTS idx_products_category
    ON products(category);

CREATE INDEX IF NOT EXISTS idx_products_sub_category
    ON products(sub_category);


-- Fleet
CREATE INDEX IF NOT EXISTS idx_fleet_vehicle_type
    ON fleet_vehicles(vehicle_type);

CREATE INDEX IF NOT EXISTS idx_fleet_hub
    ON fleet_vehicles(hub_code);

CREATE INDEX IF NOT EXISTS idx_fleet_status
    ON fleet_vehicles(fleet_status);


-- Orders
CREATE INDEX IF NOT EXISTS idx_orders_customer_id
    ON orders(customer_id);

CREATE INDEX IF NOT EXISTS idx_orders_product_id
    ON orders(product_id);

CREATE INDEX IF NOT EXISTS idx_orders_vehicle_id
    ON orders(assigned_vehicle_id);

CREATE INDEX IF NOT EXISTS idx_orders_order_date
    ON orders(order_date);

CREATE INDEX IF NOT EXISTS idx_orders_status
    ON orders(order_status);


-- Delivery Logs
CREATE INDEX IF NOT EXISTS idx_delivery_logs_order_id
    ON delivery_logs(order_id);

CREATE INDEX IF NOT EXISTS idx_delivery_logs_vehicle_id
    ON delivery_logs(vehicle_id);

CREATE INDEX IF NOT EXISTS idx_delivery_logs_timestamp
    ON delivery_logs(event_timestamp);


-- Drone Telemetry
CREATE INDEX IF NOT EXISTS idx_drone_telemetry_drone_id
    ON drone_telemetry(drone_id);

CREATE INDEX IF NOT EXISTS idx_drone_telemetry_timestamp
    ON drone_telemetry(flight_timestamp);

CREATE INDEX IF NOT EXISTS idx_drone_telemetry_maintenance
    ON drone_telemetry(maintenance_required);


-- Maintenance
CREATE INDEX IF NOT EXISTS idx_maintenance_vehicle_id
    ON maintenance_history(vehicle_id);

CREATE INDEX IF NOT EXISTS idx_maintenance_service_date
    ON maintenance_history(service_date);


-- GPS Routes
CREATE INDEX IF NOT EXISTS idx_gps_routes_vehicle_id
    ON gps_routes(vehicle_id);

CREATE INDEX IF NOT EXISTS idx_gps_routes_date
    ON gps_routes(route_date);


-- Route Orders
CREATE INDEX IF NOT EXISTS idx_route_orders_order_id
    ON route_orders(order_id);


-- Route Waypoints
CREATE INDEX IF NOT EXISTS idx_route_waypoints_route_id
    ON route_waypoints(route_id);


-- Traffic
CREATE INDEX IF NOT EXISTS idx_traffic_city_date
    ON traffic_data(city, record_date);


-- Weather
CREATE INDEX IF NOT EXISTS idx_weather_city_date
    ON weather_data(city, observation_date);


-- Customer Reviews
CREATE INDEX IF NOT EXISTS idx_reviews_customer_id
    ON customer_reviews(customer_id);

CREATE INDEX IF NOT EXISTS idx_reviews_product_id
    ON customer_reviews(product_id);

CREATE INDEX IF NOT EXISTS idx_reviews_order_id
    ON customer_reviews(order_id);

CREATE INDEX IF NOT EXISTS idx_reviews_sentiment
    ON customer_reviews(sentiment_label);
