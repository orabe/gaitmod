import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SyncDirection(Enum):
    """Enumeration for sync directions."""
    DOWNLOAD = "download"
    UPLOAD = "upload"
    BIDIRECTIONAL = "bidirectional"

class SyncMethod(Enum):
    """Enumeration for sync methods."""
    SCP = "scp"
    RSYNC = "rsync"
    AUTO = "auto"  # Choose best method automatically

@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    items_synced: int
    total_items: int
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    bytes_transferred: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.items_synced / self.total_items * 100) if self.total_items > 0 else 0.0

@dataclass
class SyncConfig:
    """Configuration for a single sync operation."""
    remote_host: str
    remote_user: str
    remote_path: str
    local_path: Optional[str] = None
    files: Optional[List[str]] = None
    sync_folder: bool = False
    target_subdir: str = ""
    method: SyncMethod = SyncMethod.AUTO
    timeout: int = 300
    retries: int = 3
    retry_delay: float = 1.0
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    preserve_permissions: bool = True
    compress: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.sync_folder and not self.files:
            raise ValueError("Either 'sync_folder' must be True or 'files' must be provided")
        
        if self.sync_folder and self.files:
            logger.warning("Both 'sync_folder' and 'files' specified. Folder sync will take precedence.")

class DataSynchronizer:
    """
    A robust, flexible data synchronization class.
    
    Features:
    - Multiple sync methods (scp, rsync)
    - Retry logic with exponential backoff
    - Progress tracking and detailed logging
    - Validation and error handling
    - Support for file and folder operations
    - Bandwidth and transfer statistics
    """
    
    def __init__(self, 
                 default_timeout: int = 300,
                 default_retries: int = 3,
                 log_level: str = "INFO"):
        """
        Initialize the DataSynchronizer.
        
        Args:
            default_timeout: Default timeout for operations in seconds
            default_retries: Default number of retry attempts
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.default_timeout = default_timeout
        self.default_retries = default_retries
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def sync(self, 
             configs: Union[SyncConfig, List[SyncConfig]], 
             target_base_path: str,
             direction: Union[SyncDirection, str] = SyncDirection.DOWNLOAD,
             force_sync: bool = False,
             dry_run: bool = False,
             progress_callback: Optional[callable] = None) -> SyncResult:
        """
        Main sync method with comprehensive error handling and validation.
        
        Args:
            configs: Single config or list of sync configurations
            target_base_path: Base directory for local operations
            direction: Sync direction (download, upload, bidirectional)
            force_sync: Force sync even if files exist
            dry_run: Show what would be synced without actually syncing
            progress_callback: Optional callback for progress updates
            
        Returns:
            SyncResult object with detailed operation results
        """
        start_time = time.time()
        
        # Normalize inputs
        if isinstance(configs, SyncConfig):
            configs = [configs]
        
        if isinstance(direction, str):
            direction = SyncDirection(direction.lower())
        
        # Validate inputs
        self._validate_inputs(configs, target_base_path, direction)
        
        # Initialize result tracking
        total_items = len(configs)
        successful_items = 0
        all_errors = []
        total_bytes = 0
        
        self.logger.info(f"Starting sync operation: {total_items} items, direction: {direction.value}")
        
        if dry_run:
            self.logger.info("DRY RUN MODE - No actual transfers will occur")
        
        # Process each configuration
        for i, config in enumerate(configs):
            try:
                self.logger.info(f"Processing item {i+1}/{total_items}: {config.remote_path}")
                
                if progress_callback:
                    progress_callback(i, total_items, config)
                
                # Execute sync operation
                item_result = self._sync_single_item(
                    config, target_base_path, direction, force_sync, dry_run
                )
                
                if item_result.success:
                    successful_items += 1
                    total_bytes += item_result.bytes_transferred
                    self.logger.info(f"Successfully synced item {i+1}")
                else:
                    all_errors.extend(item_result.errors)
                    self.logger.error(f"Failed to sync item {i+1}: {item_result.errors}")
                    
            except Exception as e:
                error_msg = f"Unexpected error processing item {i+1}: {str(e)}"
                all_errors.append(error_msg)
                self.logger.error(error_msg, exc_info=True)
        
        # Calculate final results
        duration = time.time() - start_time
        overall_success = successful_items == total_items
        
        result = SyncResult(
            success=overall_success,
            items_synced=successful_items,
            total_items=total_items,
            errors=all_errors,
            duration=duration,
            bytes_transferred=total_bytes
        )
        
        # Log summary
        self.logger.info(f"Sync completed: {result.success_rate:.1f}% success rate, "
                        f"{result.duration:.2f}s duration, {result.bytes_transferred:,} bytes")
        
        if progress_callback:
            progress_callback(total_items, total_items, None)  # Signal completion
        
        return result
    
    def _validate_inputs(self, configs: List[SyncConfig], target_base_path: str, direction: SyncDirection):
        """Validate input parameters."""
        if not configs:
            raise ValueError("At least one sync configuration must be provided")
        
        if not target_base_path and direction in [SyncDirection.DOWNLOAD, SyncDirection.BIDIRECTIONAL]:
            raise ValueError("target_base_path is required for download operations")
        
        # Validate each configuration
        for i, config in enumerate(configs):
            if direction in [SyncDirection.UPLOAD, SyncDirection.BIDIRECTIONAL] and not config.local_path:
                raise ValueError(f"Config {i}: local_path is required for upload operations")
    
    def _sync_single_item(self, 
                         config: SyncConfig, 
                         target_base_path: str, 
                         direction: SyncDirection,
                         force_sync: bool,
                         dry_run: bool) -> SyncResult:
        """Sync a single item with retry logic."""
        
        for attempt in range(config.retries + 1):
            try:
                if config.sync_folder:
                    return self._sync_folder_with_retry(
                        config, target_base_path, direction, force_sync, dry_run, attempt
                    )
                else:
                    return self._sync_files_with_retry(
                        config, target_base_path, direction, force_sync, dry_run, attempt
                    )
                    
            except subprocess.TimeoutExpired:
                if attempt < config.retries:
                    delay = config.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {delay}s")
                    time.sleep(delay)
                else:
                    return SyncResult(False, 0, 1, ["Operation timed out after all retries"])
                    
            except Exception as e:
                if attempt < config.retries:
                    delay = config.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Error on attempt {attempt + 1}: {e}, retrying in {delay}s")
                    time.sleep(delay)
                else:
                    return SyncResult(False, 0, 1, [f"Failed after {config.retries + 1} attempts: {str(e)}"])
        
        return SyncResult(False, 0, 1, ["Max retries exceeded"])
    
    def _sync_folder_with_retry(self, config: SyncConfig, target_base_path: str, 
                               direction: SyncDirection, force_sync: bool, dry_run: bool, attempt: int) -> SyncResult:
        """Sync entire folder using rsync."""
        
        # Determine paths
        if direction == SyncDirection.DOWNLOAD:
            target_dir = os.path.join(target_base_path, config.target_subdir) if config.target_subdir else target_base_path
            os.makedirs(target_dir, exist_ok=True)
            source = f"{config.remote_user}@{config.remote_host}:{config.remote_path}/"
            destination = target_dir
        else:  # UPLOAD
            source = f"{config.local_path}/"
            destination = f"{config.remote_user}@{config.remote_host}:{config.remote_path}/"
        
        # Build rsync command
        cmd = self._build_rsync_command(config, source, destination, force_sync, dry_run)
        
        self.logger.debug(f"Executing: {' '.join(cmd)}")
        
        # Execute command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout
        )
        
        if result.returncode == 0:
            # Parse transferred bytes from rsync output (simplified)
            bytes_transferred = self._parse_rsync_output(result.stdout)
            return SyncResult(True, 1, 1, [], bytes_transferred=bytes_transferred)
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown rsync error"
            return SyncResult(False, 0, 1, [error_msg])
    
    def _sync_files_with_retry(self, config: SyncConfig, target_base_path: str,
                              direction: SyncDirection, force_sync: bool, dry_run: bool, attempt: int) -> SyncResult:
        """Sync individual files."""
        
        successful_files = 0
        errors = []
        total_bytes = 0
        
        # Determine target directory for downloads
        if direction == SyncDirection.DOWNLOAD:
            target_dir = os.path.join(target_base_path, config.target_subdir) if config.target_subdir else target_base_path
            os.makedirs(target_dir, exist_ok=True)
        
        for filename in config.files:
            try:
                if direction == SyncDirection.DOWNLOAD:
                    success, bytes_transferred = self._download_file(config, target_dir, filename, force_sync, dry_run)
                else:  # UPLOAD
                    success, bytes_transferred = self._upload_file(config, filename, force_sync, dry_run)
                
                if success:
                    successful_files += 1
                    total_bytes += bytes_transferred
                else:
                    errors.append(f"Failed to sync {filename}")
                    
            except Exception as e:
                errors.append(f"Error syncing {filename}: {str(e)}")
        
        success = successful_files == len(config.files)
        return SyncResult(success, successful_files, len(config.files), errors, bytes_transferred=total_bytes)
    
    def _download_file(self, config: SyncConfig, target_dir: str, filename: str, 
                      force_sync: bool, dry_run: bool) -> Tuple[bool, int]:
        """Download a single file."""
        
        remote_path = f"{config.remote_user}@{config.remote_host}:{config.remote_path}/{filename}"
        local_path = os.path.join(target_dir, filename)
        
        # Check if file exists
        if not force_sync and os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            self.logger.info(f"  {filename} already exists locally ({file_size:,} bytes)")
            return True, file_size
        
        if dry_run:
            self.logger.info(f"  Would download {filename}")
            return True, 0
        
        # Choose sync method
        if config.method == SyncMethod.AUTO:
            method = SyncMethod.SCP  # Default to SCP for single files
        else:
            method = config.method
        
        if method == SyncMethod.SCP:
            cmd = ["scp", remote_path, local_path]
        else:  # RSYNC
            cmd = ["rsync", "-avz", remote_path, local_path]
        
        self.logger.debug(f"Downloading {filename} using {method.value}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.timeout)
        
        if result.returncode == 0 and os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            self.logger.info(f"  Downloaded {filename} ({file_size:,} bytes)")
            return True, file_size
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            self.logger.error(f"  Failed to download {filename}: {error_msg}")
            return False, 0
    
    def _upload_file(self, config: SyncConfig, filename: str, force_sync: bool, dry_run: bool) -> Tuple[bool, int]:
        """Upload a single file."""
        
        local_path = os.path.join(config.local_path, filename)
        remote_path = f"{config.remote_user}@{config.remote_host}:{config.remote_path}/{filename}"
        
        if not os.path.exists(local_path):
            self.logger.error(f"  Local file {filename} not found")
            return False, 0
        
        file_size = os.path.getsize(local_path)
        
        if dry_run:
            self.logger.info(f"  Would upload {filename} ({file_size:,} bytes)")
            return True, file_size
        
        # Choose sync method
        if config.method == SyncMethod.AUTO:
            method = SyncMethod.SCP
        else:
            method = config.method
        
        if method == SyncMethod.SCP:
            cmd = ["scp", local_path, remote_path]
        else:  # RSYNC
            cmd = ["rsync", "-avz", local_path, remote_path]
        
        self.logger.debug(f"Uploading {filename} using {method.value}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.timeout)
        
        if result.returncode == 0:
            self.logger.info(f"  Uploaded {filename} ({file_size:,} bytes)")
            return True, file_size
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            self.logger.error(f"  Failed to upload {filename}: {error_msg}")
            return False, 0
    
    def _build_rsync_command(self, config: SyncConfig, source: str, destination: str, 
                           force_sync: bool, dry_run: bool) -> List[str]:
        """Build rsync command with all options."""
        
        cmd = ["rsync"]
        
        # Basic options
        if config.preserve_permissions:
            cmd.append("-a")  # Archive mode (preserves permissions, timestamps, etc.)
        if config.compress:
            cmd.append("-z")  # Compress
        cmd.append("-v")  # Verbose
        cmd.append("--progress")  # Show progress
        
        # Conditional options
        if not force_sync:
            cmd.append("--update")  # Skip files that are newer on destination
        if dry_run:
            cmd.append("--dry-run")
        
        # Include/exclude patterns
        for pattern in config.exclude_patterns:
            cmd.extend(["--exclude", pattern])
        for pattern in config.include_patterns:
            cmd.extend(["--include", pattern])
        
        # Source and destination
        cmd.extend([source, destination])
        
        return cmd
    
    def _parse_rsync_output(self, output: str) -> int:
        """Parse rsync output to extract bytes transferred."""
        # Simplified parser - look for "sent X bytes" line
        for line in output.split('\n'):
            if 'sent' in line and 'bytes' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'sent' and i + 1 < len(parts):
                        try:
                            return int(parts[i + 1].replace(',', ''))
                        except ValueError:
                            pass
        return 0

# Convenience functions for backward compatibility
def sync_data(source_configs, target_base_path, direction='download', force_sync=False, verbose=True):
    """
    Backward-compatible wrapper for the DataSynchronizer class.
    
    This maintains the original API while providing access to the new functionality.
    """
    
    # Convert old-style configs to new SyncConfig objects
    if isinstance(source_configs, dict):
        source_configs = [source_configs]
    
    sync_configs = []
    for config in source_configs:
        sync_config = SyncConfig(
            remote_host=config['remote_host'],
            remote_user=config['remote_user'],
            remote_path=config['remote_path'],
            local_path=config.get('local_path'),
            files=config.get('files'),
            sync_folder=config.get('sync_folder', False),
            target_subdir=config.get('target_subdir', ''),
        )
        sync_configs.append(sync_config)
    
    # Create synchronizer
    log_level = "INFO" if verbose else "WARNING"
    synchronizer = DataSynchronizer(log_level=log_level)
    
    # Perform sync
    result = synchronizer.sync(
        configs=sync_configs,
        target_base_path=target_base_path,
        direction=direction,
        force_sync=force_sync
    )
    
    return result.success

# Example usage and factory functions
def create_hctsa_sync_configs(local_base_path: str) -> List[SyncConfig]:
    """Factory function to create HCTSA sync configurations."""
    return [
        SyncConfig(
            remote_host='141.23.1.143',
            remote_user='orabem',
            remote_path='/home/orabem/hctsa',
            local_path=local_base_path,
            files=['HCTSA.mat', 'HCTSA_N.mat'],
            timeout=600,  # Longer timeout for large files
        ),
        SyncConfig(
            remote_host='141.23.1.143',
            remote_user='orabem',
            remote_path='/home/orabem/hctsa/data/hctsa_output_data',
            local_path=os.path.join(local_base_path, 'data', 'hctsa_output_data'),
            files=['Operations.csv', 'TimeSeries.csv', 'MasterOperations.csv'],
            target_subdir='data/hctsa_output_data',
        )
    ]

def sync_hctsa_data(local_base_path: str, direction: str = 'download', 
                   force_sync: bool = False, progress_callback: Optional[callable] = None) -> SyncResult:
    """Convenience function for HCTSA data synchronization."""
    
    configs = create_hctsa_sync_configs(local_base_path)
    synchronizer = DataSynchronizer()
    
    return synchronizer.sync(
        configs=configs,
        target_base_path=local_base_path,
        direction=direction,
        force_sync=force_sync,
        progress_callback=progress_callback
    )
    
    
# Modern API
synchronizer = DataSynchronizer()
configs = create_hctsa_sync_configs('/local/path')
result = synchronizer.sync(configs, '/local/path', direction='download')

# Backward compatible API
success = sync_data(old_style_configs, '/local/path', direction='download')

# With progress tracking
def progress_handler(current, total, config):
    print(f"Progress: {current}/{total}")

result = sync_hctsa_data('/local/path', progress_callback=progress_handler)