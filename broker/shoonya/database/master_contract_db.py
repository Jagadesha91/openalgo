import io
import os
import zipfile
from datetime import datetime

import pandas as pd
import requests
from sqlalchemy import Column, Float, Index, Integer, Sequence, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

from database.engine_factory import create_db_engine
from extensions import socketio
from utils.logging import get_logger

logger = get_logger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_db_engine(DATABASE_URL)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()


class SymToken(Base):
    __tablename__ = "symtoken"
    id = Column(Integer, Sequence("symtoken_id_seq"), primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    brsymbol = Column(String, nullable=False, index=True)
    name = Column(String)
    exchange = Column(String, index=True)
    brexchange = Column(String, index=True)
    token = Column(String, index=True)
    expiry = Column(String)
    strike = Column(Float)
    lotsize = Column(Integer)
    instrumenttype = Column(String)
    tick_size = Column(Float)

    __table_args__ = (Index("idx_symbol_exchange", "symbol", "exchange"),)


def init_db():
    logger.info("Initializing Master Contract DB")
    Base.metadata.create_all(bind=engine)


def delete_symtoken_table():
    logger.info("Deleting Symtoken Table")
    SymToken.query.delete()
    db_session.commit()


def drop_incomplete_rows(df, exchange, required=("brsymbol", "name")):
    subset = [column for column in required if column in df.columns]
    if not subset:
        return df

    cleaned = df.dropna(subset=subset)
    dropped = len(df) - len(cleaned)
    if dropped:
        logger.warning(
            f"Skipping {dropped} incomplete {exchange} rows with blank {'/'.join(subset)}"
        )
    return cleaned.copy()


def replace_symtoken_table(frames):
    records = []
    seen = set()
    for segment, df in frames:
        if df is None or df.empty:
            raise RuntimeError(
                f"The Shoonya {segment} symbol file had no usable rows. "
                "Your existing symbols were kept; try the download again."
            )
        df = df.dropna(subset=["symbol", "brsymbol"])
        rows = [
            row
            for row in df.to_dict(orient="records")
            if (row["token"], row["exchange"]) not in seen
        ]
        seen.update((row["token"], row["exchange"]) for row in rows)
        records.extend(rows)

    try:
        SymToken.query.delete()
        db_session.bulk_insert_mappings(SymToken, records)
        db_session.commit()
        logger.info(f"Replaced symtoken table with {len(records)} NIFTY option records")
    except Exception:
        db_session.rollback()
        raise


# Download ONLY NFO symbols to keep memory usage minimal
shoonya_urls = {
    "NFO": "https://api.shoonya.com/NFO_symbols.txt.zip",
}


def download_and_unzip_shoonya_data(output_path):
    logger.info("Downloading and Unzipping Shoonya NFO Data")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    delete_shoonya_temp_data(output_path)

    downloaded_files = []
    failed = []

    for key, url in shoonya_urls.items():
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                logger.info(f"Successfully downloaded {key} from {url}")
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(output_path)
                downloaded_files.append(f"{key}.txt")
            else:
                logger.error(
                    f"Failed to download {key} from {url}. Status code: {response.status_code}"
                )
                failed.append(key)
        except Exception as e:
            logger.error(f"Error downloading {key} from {url}: {e}")
            failed.append(key)

    if failed:
        raise RuntimeError(
            f"Could not download the Shoonya symbol files for {', '.join(failed)}. "
            "Your existing symbols were kept; try the download again."
        )

    return downloaded_files


def process_shoonya_nfo_data(output_path):
    """
    Processes Shoonya NFO data (NFO_symbols.txt) and filters strictly for NIFTY Options (CE/PE).
    """
    logger.info("Processing Shoonya NFO Data (NIFTY Options Only)")
    file_path = f"{output_path}/NFO_symbols.txt"

    df = pd.read_csv(
        file_path,
        usecols=[
            "Exchange",
            "Token",
            "LotSize",
            "Symbol",
            "TradingSymbol",
            "Expiry",
            "Instrument",
            "OptionType",
            "StrikePrice",
            "TickSize",
        ],
    )

    df.columns = [
        "exchange",
        "token",
        "lotsize",
        "name",
        "brsymbol",
        "expiry",
        "instrumenttype",
        "optiontype",
        "strike",
        "tick_size",
    ]

    df = drop_incomplete_rows(df, "NFO")

    # FILTER STRICTLY FOR NIFTY OPTIONS (CE and PE only; excludes futures & other symbols)
    df = df[
        (df["name"] == "NIFTY")
        & (df["instrumenttype"] == "OPTIDX")
        & (df["optiontype"].isin(["CE", "PE"]))
    ].copy()

    df["expiry"] = df["expiry"].fillna("")
    df["strike"] = df["strike"].fillna("-1")

    def format_expiry_date(date_str):
        try:
            date_obj = datetime.strptime(date_str, "%d-%b-%Y")
            return date_obj.strftime("%d-%b-%y").upper()
        except ValueError:
            logger.info(f"Invalid expiry date format: {date_str}")
            return None

    df["expiry"] = df["expiry"].apply(format_expiry_date)
    df["instrumenttype"] = df["optiontype"]

    def format_symbol(row):
        expiry_date = row["expiry"]
        compact_expiry = expiry_date.replace("-", "") if (expiry_date and isinstance(expiry_date, str)) else ""
        formatted_strike = (
            int(row["strike"]) if float(row["strike"]).is_integer() else row["strike"]
        )
        return f"{row['name']}{compact_expiry}{formatted_strike}{row['instrumenttype']}"

    df["symbol"] = df.apply(format_symbol, axis=1)
    df["exchange"] = "NFO"
    df["brexchange"] = "NFO"

    def handle_strike_price(strike):
        try:
            if float(strike).is_integer():
                return int(float(strike))
            else:
                return float(strike)
        except (ValueError, TypeError):
            return -1

    df["strike"] = df["strike"].apply(handle_strike_price)

    columns_to_keep = [
        "symbol",
        "brsymbol",
        "name",
        "exchange",
        "brexchange",
        "token",
        "expiry",
        "strike",
        "lotsize",
        "instrumenttype",
        "tick_size",
    ]
    return df[columns_to_keep]


def delete_shoonya_temp_data(output_path):
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        if filename.endswith(".txt") and os.path.isfile(file_path):
            os.remove(file_path)
            logger.info(f"Deleted {file_path}")


def master_contract_download():
    logger.info("Downloading Shoonya Master Contract (NIFTY Options Only)")

    output_path = os.path.join("tmp", "shoonya")
    try:
        download_and_unzip_shoonya_data(output_path)

        frames = [
            ("NFO", process_shoonya_nfo_data(output_path)),
        ]
        replace_symtoken_table(frames)

        delete_shoonya_temp_data(output_path)

        return socketio.emit(
            "master_contract_download",
            {"status": "success", "message": "Successfully Downloaded NIFTY Options"},
        )
    except Exception as e:
        logger.exception(f"Shoonya master contract download failed: {e}")
        socketio.emit("master_contract_download", {"status": "error", "message": str(e)})
        raise
