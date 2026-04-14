import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime


class CarPriceModel:
    """Random Forest model for India used car price prediction."""

    PRICE_KEYWORDS = ["selling_price", "price", "cost", "value", "sell_price"]
    YEAR_KEYWORDS  = ["year", "yr", "model_year"]
    KM_KEYWORDS    = ["km", "kms", "driven", "mileage", "odometer"]
    FUEL_KEYWORDS  = ["fuel", "fuel_type", "petrol", "diesel"]
    TRANS_KEYWORDS = ["transmission", "trans", "gear"]
    OWNER_KEYWORDS = ["owner", "ownership"]
    BRAND_KEYWORDS = ["brand", "make", "manufacturer", "company"]
    NAME_KEYWORDS  = ["name", "car_name"]   # removed "model" to avoid matching Model column

    def __init__(self, n_estimators=200, max_depth=10, test_size=0.2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.test_size    = test_size
        self.random_state = random_state

        self.model        = None
        self.encoders     = {}
        self.feature_cols = []
        self.target_col   = None
        self.feature_info = {}
        self.metrics      = {}
        self.y_test       = None
        self.y_pred       = None

    # ─── Helpers ────────────────────────────────────────────────────
    def _find_col(self, df, keywords):
        cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}
        for kw in keywords:
            for lower_col, orig_col in cols_lower.items():
                if kw in lower_col:
                    return orig_col
        return None

    def _is_non_numeric(self, series):
        """True for object, string, category — anything not purely numeric."""
        return not pd.api.types.is_numeric_dtype(series)

    def _get_non_numeric_cols(self, df, exclude=None):
        """Return all non-numeric column names, optionally excluding some."""
        exclude = exclude or []
        return [c for c in df.columns if c not in exclude and self._is_non_numeric(df[c])]

    def _detect_target(self, df):
        # First pass: keyword match on numeric columns
        for kw in self.PRICE_KEYWORDS:
            col = self._find_col(df, [kw])
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return col
        # Second pass: keyword match even if column is object dtype
        # (handles case where currency cleaning wasn't applied yet)
        for kw in self.PRICE_KEYWORDS:
            col = self._find_col(df, [kw])
            if col:
                return col
        # Fallback: numeric column with highest mean, explicitly excluding
        # year-like columns (values between 1900-2100) and Id
        num_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if c.lower() not in ("id",)
            and not any(kw in c.lower() for kw in self.YEAR_KEYWORDS)
            and not (df[c].dropna().between(1900, 2100).all() and df[c].nunique() < 50)
        ]
        if num_cols:
            return max(num_cols, key=lambda c: df[c].mean())
        return None

    # ─── Currency Cleaning ───────────────────────────────────────────
    def _clean_currency_cols(self, df):
        """Detect columns with ₹ / currency strings and convert to numeric lakhs.
        Handles Windows encoding issues where ₹ may appear as garbled bytes."""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(20)
                # Match ₹ literally, its common mis-encodings (â‚¹), Rs prefix, or comma-numbers
                has_currency = (
                    sample.str.contains(r"[\u20b9\â‚¹,]", regex=True).any()
                    or sample.str.contains(r"Rs\.?\s*\d", regex=True).any()
                    or sample.str.contains(r"^\s*[\d,]+\s*$", regex=True).any()
                )
                if has_currency:
                    cleaned = (
                        df[col].astype(str)
                        # Strip ₹ (U+20B9), mojibake variants, Rs prefix
                        .str.replace(u"\u20b9", "", regex=False)
                        .str.replace(r"â‚¹|Rs\.?", "", regex=True)
                        # Strip commas and whitespace
                        .str.replace(r"[,\s]", "", regex=True)
                        .str.strip()
                        # Remove any remaining non-numeric chars (keep dot/minus)
                        .str.replace(r"[^\d.\-]", "", regex=True)
                    )
                    numeric = pd.to_numeric(cleaned, errors="coerce")
                    if numeric.dropna().empty:
                        continue
                    # Values > 1000 are raw rupees → convert to lakhs
                    if numeric.median() > 1000:
                        numeric = numeric / 1_00_000
                    df[col] = numeric
        return df

    # ─── KM Cleaning ─────────────────────────────────────────────────
    def _clean_km_cols(self, df):
        """Strip 'km' suffix and commas from driven columns."""
        df = df.copy()
        km_col = self._find_col(df, self.KM_KEYWORDS)
        if km_col and df[km_col].dtype == object:
            df[km_col] = (
                df[km_col].astype(str)
                .str.replace(r"[,\s]*(km|kms|kilometre)s?", "", regex=True, flags=__import__("re").IGNORECASE)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[km_col] = pd.to_numeric(df[km_col], errors="coerce")
        return df

    # ─── Preprocessing ──────────────────────────────────────────────
    def _preprocess(self, df, fit=True):
        df = df.copy()

        # Clean currency and KM columns first
        df = self._clean_currency_cols(df)
        df = self._clean_km_cols(df)

        # Drop Id column
        df.drop(columns=["Id"], errors="ignore", inplace=True)

        # Convert Car_Age from year column
        year_col = self._find_col(df, self.YEAR_KEYWORDS)
        if year_col:
            df["Car_Age"] = datetime.now().year - pd.to_numeric(df[year_col], errors="coerce")
            df.drop(columns=[year_col], inplace=True)

        # Extract brand from high-cardinality name column
        name_col = self._find_col(df, self.NAME_KEYWORDS)
        if name_col and name_col in df.columns and df[name_col].nunique() > 50:
            df["Brand"] = df[name_col].astype(str).str.split().str[0]
            df.drop(columns=[name_col], inplace=True)

        # ── Encode ALL non-numeric columns (excluding target) ────────
        non_num_cols = self._get_non_numeric_cols(
            df, exclude=[self.target_col] if self.target_col else []
        )

        for col in non_num_cols:
            df[col] = df[col].astype(str).str.strip()
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le
            else:
                if col in self.encoders:
                    le = self.encoders[col]
                    df[col] = df[col].map(
                        lambda x, _le=le: int(_le.transform([x])[0]) if x in _le.classes_ else 0
                    )
                else:
                    df[col] = 0

        # Safety net: force any remaining non-numeric columns to 0
        for col in df.columns:
            if col == self.target_col:
                continue
            if self._is_non_numeric(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Fill numeric nulls with median (skip target to avoid distortion)
        for col in df.select_dtypes(include=np.number).columns:
            if col == self.target_col:
                continue
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        return df

    # ─── Feature Info (for UI form) ─────────────────────────────────
    def _build_feature_info(self, df_raw):
        df = df_raw.drop(columns=[self.target_col, "Id"], errors="ignore")
        info = {}

        year_col = self._find_col(df, self.YEAR_KEYWORDS)
        name_col = self._find_col(df, self.NAME_KEYWORDS)
        drop_cols = []

        if year_col:
            drop_cols.append(year_col)
            info["Car_Age"] = {"type": "numeric", "min": 0, "max": 30, "median": 5}
        if name_col and name_col in df.columns and df[name_col].nunique() > 50:
            drop_cols.append(name_col)

        for col in df.columns:
            if col in drop_cols:
                continue
            if self._is_non_numeric(df[col]) or df[col].nunique() <= 15:
                info[col] = {
                    "type":    "categorical",
                    "options": sorted(df[col].dropna().astype(str).unique().tolist()),
                }
            else:
                info[col] = {
                    "type":   "numeric",
                    "min":    float(df[col].min()),
                    "max":    float(df[col].max()),
                    "median": float(df[col].median()),
                }
        return info

    # ─── Train ──────────────────────────────────────────────────────
    def train(self, df_raw):
        try:
            # Clean currency/km columns before target detection
            df_raw = self._clean_currency_cols(df_raw)
            df_raw = self._clean_km_cols(df_raw)

            # Detect target first on raw data
            target = self._detect_target(df_raw)
            if target is None:
                return False, "Could not detect price/target column."

            self.target_col = target
            df = df_raw.copy().dropna(subset=[self.target_col])

            self.feature_info = self._build_feature_info(df)
            df_processed = self._preprocess(df, fit=True)

            if self.target_col not in df_processed.columns:
                return False, (
                    f"Target column '{self.target_col}' was lost during preprocessing. "
                    f"Available columns: {list(df_processed.columns)}"
                )

            X = df_processed.drop(columns=[self.target_col], errors="ignore")
            y = df_processed[self.target_col]

            # ── Final safety net: ensure target is truly numeric ────────
            if not pd.api.types.is_numeric_dtype(y):
                y = (
                    y.astype(str)
                    .str.replace(u"\u20b9", "", regex=False)   # ₹
                    .str.replace(r"â‚¹|Rs\.?", "", regex=True)  # mojibake / Rs
                    .str.replace(r"[,\s]", "", regex=True)
                    .str.replace(r"[^\d.\-]", "", regex=True)
                )
                y = pd.to_numeric(y, errors="coerce")
                if y.median() > 1000:
                    y = y / 1_00_000

            mask = y.notna()
            X, y = X[mask], y[mask]

            self.feature_cols = X.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model.fit(X_train, y_train)

            y_pred_train = self.model.predict(X_train)
            y_pred_test  = self.model.predict(X_test)

            self.y_test = y_test.values
            self.y_pred = y_pred_test

            self.metrics = {
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2":  r2_score(y_test,  y_pred_test),
                "rmse":     np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "mae":      mean_absolute_error(y_test, y_pred_test),
            }
            return True, "Model trained successfully!"

        except Exception as e:
            import traceback
            return False, f"{str(e)}\n{traceback.format_exc()}"

    # ─── Predict ────────────────────────────────────────────────────
    def predict(self, input_df):
        """Returns (predicted_price, lower_bound, upper_bound)."""
        processed = self._preprocess_input(input_df)
        X = processed.reindex(columns=self.feature_cols, fill_value=0)

        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        price = float(np.mean(tree_preds))
        std   = float(np.std(tree_preds))
        return price, max(0, price - 1.96 * std), price + 1.96 * std

    def _preprocess_input(self, input_df):
        df = input_df.copy()

        # Handle year → Car_Age
        year_col = self._find_col(df, self.YEAR_KEYWORDS)
        if year_col:
            df["Car_Age"] = datetime.now().year - pd.to_numeric(df[year_col], errors="coerce")
            df.drop(columns=[year_col], inplace=True)

        # Encode using saved encoders
        for col in list(df.columns):
            if col in self.encoders:
                le = self.encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x, _le=le: int(_le.transform([x])[0]) if x in _le.classes_ else 0
                )

        return df.reindex(columns=self.feature_cols, fill_value=0)

    # ─── Feature Importance ─────────────────────────────────────────
    def get_feature_importance(self):
        if self.model is None:
            return None
        return pd.DataFrame({
            "Feature":    self.feature_cols,
            "Importance": self.model.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)