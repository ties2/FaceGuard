from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"

    # --- Database -----------------------------------------------------------
    # DATABASE_URL wins when set (e.g. sqlite:///./faceguard.db for local runs and
    # CI). Otherwise the PostgreSQL settings below are used, exactly as in MSP.
    database_url: str | None = None
    postgres_user: str = "admin"
    postgres_password: str = "secretpassword"
    postgres_db: str = "ml_serving_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # --- Artifact storage (MSP) ----------------------------------------------
    ARTIFACT_STORAGE_BACKEND: str = "local"
    LOCAL_ARTIFACT_DIR: str = "/app/model_artifacts/"

    # --- FaceGuard ------------------------------------------------------------
    # Secret used to HMAC-sign liveness session tokens and challenge nonces.
    faceguard_session_secret: str = "dev-only-change-me"
    # Decision policy (thresholds) produced by the calibration step.
    faceguard_policy_path: str = "configs/policy.yaml"
    # Vendor adapter: "simulated" (offline demo) or "facetec" (FaceTec Server).
    faceguard_vendor: str = "simulated"
    facetec_server_url: str | None = None
    facetec_device_key: str | None = None
    # Face detector / recogniser weights (OpenCV Zoo, MIT / Apache-2.0).
    faceguard_yunet_path: str = "assets/models/face_detection_yunet_2023mar.onnx"
    faceguard_sface_path: str = "assets/models/face_recognition_sface_2021dec.onnx"

    gf_security_admin_user: str | None = None
    gf_security_admin_password: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
