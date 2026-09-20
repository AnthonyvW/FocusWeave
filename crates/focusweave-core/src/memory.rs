//! How much memory the machine can spare, for sizing the worker pool.

/// Physical memory currently available, in bytes, or `None` when this platform
/// has no cheap way to ask.
pub fn available_bytes() -> Option<u64> {
    platform::available_bytes()
}

#[cfg(target_os = "linux")]
mod platform {
    pub fn available_bytes() -> Option<u64> {
        // MemAvailable is the kernel's own estimate of what can be handed out
        // without swapping, which is exactly the question being asked.
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
}

#[cfg(windows)]
mod platform {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub fn available_bytes() -> Option<u64> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: the struct is correctly sized and laid out per the Win32
        // definition, and `length` is set as the API requires.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok == 0 {
            return None;
        }
        Some(status.avail_phys)
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod platform {
    pub fn available_bytes() -> Option<u64> {
        None
    }
}
