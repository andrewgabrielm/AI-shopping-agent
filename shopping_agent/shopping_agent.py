import base64
import json
import os
import sqlite3
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")


# Use an authenticated user ID in a multi-user application.
# This default is only for a single-user demo.
DEFAULT_USER_ID = os.getenv("SHOP_USER_ID", "demo_user")

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
vision_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

# ============================================================
# DATABASE
# ============================================================

def _connect():
    """Create a SQLite connection with dictionary-like rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                product_name TEXT NOT NULL,
                price REAL NOT NULL,
                ordered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT
            )
        """)

        # Check the columns in an existing orders table.
        cursor.execute("PRAGMA table_info(orders)")
        columns = {row[1] for row in cursor.fetchall()}

        if "user_id" not in columns:
            cursor.execute(
                "ALTER TABLE orders ADD COLUMN user_id TEXT"
            )

        if "ordered_at" not in columns:
            cursor.execute(
                "ALTER TABLE orders ADD COLUMN ordered_at TEXT"
            )

        # Fill in missing dates and associate old orders
        # with the single demo user.
        cursor.execute("""
            UPDATE orders
            SET ordered_at = CURRENT_TIMESTAMP
            WHERE ordered_at IS NULL OR ordered_at = ''
        """)

        cursor.execute("""
            UPDATE orders
            SET user_id = ?
            WHERE user_id IS NULL OR user_id = ''
        """, (DEFAULT_USER_ID,))

        # Create persistent user preferences.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                max_price REAL,
                is_organic INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_user_date
            ON orders(user_id, ordered_at)
        """)

        conn.commit()

    finally:
        conn.close()

def _read_user_preferences():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT max_price, is_organic
            FROM user_preferences
            WHERE user_id = ?
        """, (DEFAULT_USER_ID,))

        row = cursor.fetchone()

        if row is None:
            return {
                "max_price": None,
                "is_organic": None
            }

        return {
            "max_price": row[0],
            "is_organic": (
                None if row[1] is None else bool(row[1])
            )
        }

    finally:
        conn.close()


# Initialize persistent storage when the application starts.
init_db()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None, use_saved_preferences: bool = True,) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic.
    """
    if use_saved_preferences:
        preferences = _read_user_preferences()

        if max_price is None:
            max_price = preferences["max_price"]

        if is_organic is None:
            is_organic = preferences["is_organic"]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params: list = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id":          row[0],
            "name":        row[1],
            "category":    row[2],
            "price":       row[3],
            "description": row[4],
            "is_organic":  bool(row[5]),
        }
        for row in rows
    ]
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and returns
    a confirmation message with the order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?, ?)",
        (product_id, name, price, DEFAULT_USER_ID),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract its key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
                "- search_query: a short keyword to search for it (e.g. 'honey', 'olive oil')\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return response.content

@tool
def order_history(query: str = "", limit: int = 20) -> str:
    """
    Retrieve the user's previous orders.
    Search by product name or return recent orders.
    """

    limit = max(1, min(int(limit), 100))
    search_term = query.strip()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        sql = """
            SELECT id, product_id, product_name, price, ordered_at
            FROM orders
            WHERE user_id = ?
        """
        params = [DEFAULT_USER_ID]

        if search_term.lower() not in {
            "", "all", "all orders", "order history", "my orders"
        }:
            sql += " AND product_name LIKE ?"
            params.append(f"%{search_term}%")

        sql += " ORDER BY ordered_at DESC, id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        orders = [
            {
                "order_id": row[0],
                "product_id": row[1],
                "product_name": row[2],
                "price": row[3],
                "ordered_at": row[4],
            }
            for row in rows
        ]

        return json.dumps({
            "count": len(orders),
            "orders": orders,
        })

    finally:
        conn.close()

@tool
def get_user_preferences() -> str:
    """Retrieve the user's saved shopping preferences."""

    return json.dumps(_read_user_preferences())


@tool
def update_user_preferences(
    is_organic: Optional[bool] = None,
    max_price: Optional[float] = None,
    clear_fields: Optional[list[str]] = None,
) -> str:
    """
    Save or update persistent shopping preferences.

    is_organic=True means prefer organic products.
    is_organic=False means prefer non-organic products.
    max_price sets the maximum preferred price.
    Use clear_fields to remove 'is_organic' or 'max_price'.
    """

    clear_fields = clear_fields or []
    allowed_fields = {"is_organic", "max_price"}

    if set(clear_fields) - allowed_fields:
        return json.dumps({
            "error": "Invalid preference field to clear."
        })

    if max_price is not None and max_price < 0:
        return json.dumps({
            "error": "Maximum price cannot be negative."
        })

    current = _read_user_preferences()

    saved_max_price = current["max_price"]
    saved_is_organic = current["is_organic"]

    if "max_price" in clear_fields:
        saved_max_price = None
    elif max_price is not None:
        saved_max_price = float(max_price)

    if "is_organic" in clear_fields:
        saved_is_organic = None
    elif is_organic is not None:
        saved_is_organic = is_organic

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT user_id FROM user_preferences WHERE user_id = ?",
            (DEFAULT_USER_ID,)
        )

        existing = cursor.fetchone()

        if existing:
            cursor.execute("""
                UPDATE user_preferences
                SET max_price = ?,
                    is_organic = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (
                saved_max_price,
                (
                    None
                    if saved_is_organic is None
                    else int(saved_is_organic)
                ),
                DEFAULT_USER_ID,
            ))
        else:
            cursor.execute("""
                INSERT INTO user_preferences
                    (user_id, max_price, is_organic)
                VALUES (?, ?, ?)
            """, (
                DEFAULT_USER_ID,
                saved_max_price,
                (
                    None
                    if saved_is_organic is None
                    else int(saved_is_organic)
                ),
            ))

        conn.commit()

        return json.dumps({
            "message": "Preferences saved successfully.",
            "preferences": {
                "max_price": saved_max_price,
                "is_organic": saved_is_organic,
            },
        })

    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_agent(
    llm,
    tools=[
    search_products,
    get_rating,
    checkout,
    describe_product_image,
    order_history,
    get_user_preferences,
    update_user_preferences,
    ],
    system_prompt=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "MEMORY AND PERSONALIZATION:\n"
        "1. When the user asks to remember a lasting shopping preference, "
        "call update_user_preferences to save it.\n"
        "2. Use get_user_preferences when the user asks about saved preferences.\n"
        "3. Product searches automatically apply saved preferences to filters "
        "that the user has not explicitly specified.\n"
        "4. Explicit filters in a new request override the corresponding "
        "saved preferences for that search.\n"
        "5. If the user asks to ignore saved preferences, call search_products "
        "with use_saved_preferences=False.\n"
        "6. When the user asks to remove a preference, use update_user_preferences "
        "with clear_fields.\n"
        "7. Never claim a preference was saved unless the tool confirms it.\n\n"

        "ORDER HISTORY:\n"
        "1. When the user asks what they ordered before, call order_history "
        "with an empty query.\n"
        "2. When the user asks about a particular product they bought, "
        "search the history using the product name or keyword.\n"
        "3. Summarize the actual results, including product name, order ID, "
        "date, and price where available.\n"
        "4. If no matching orders are returned, say so clearly.\n"
        "5. Never invent or assume an order exists.\n\n"
        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products.\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        "BROWSING — when the user describes what they want to buy:\n"
        "1. Call search_products to find matching items (apply any price/organic filters given).\n"
        "2. For each candidate, call get_rating to retrieve its average rating.\n"
        "3. Filter by the user's minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use this exact format "
        "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability. "
        "   Always include (ID:X) so you can reference it later.\n"
        "5. If only one product qualifies, still show it in the list and ask: "
        "   'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
        "'order number 2', 'the first one', 'get me #3'):\n"
        "1. Look at your previous message to find the (ID:X) for the chosen product "
        "   (if only one was listed and the user says 'yes', use that product's ID).\n"
        "2. Call checkout with that product_id (the number from (ID:X)).\n"
        "3. Confirm the order to the user in plain text.\n\n"
        "Never place an order unless the user explicitly confirms. "
        "Never guess a product_id — always take it from the (ID:X) in your own previous message."
    ),
    checkpointer=InMemorySaver()
    
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)
