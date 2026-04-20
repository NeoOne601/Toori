import socket
import asyncio
from zeroconf.asyncio import AsyncZeroconf
from zeroconf import ServiceInfo

class BonjourAdvertiser:
    def __init__(self, name: str = "Toori Reality Engine", port: int = 7777):
        self.name = name
        self.port = port
        self.aiozc = None
        self.info = None

    async def start(self):
        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = '127.0.0.1'
        finally:
            s.close()

        desc = {'version': '1.0.0'}
        
        self.info = ServiceInfo(
            "_toori._tcp.local.",
            f"{self.name}._toori._tcp.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties=desc,
            server=f"{socket.gethostname()}.local.",
        )
        
        self.aiozc = AsyncZeroconf()
        print(f"🚀 [Bonjour] Advertising {self.name} on {local_ip}:{self.port} (_toori._tcp)")
        await self.aiozc.zeroconf.async_register_service(self.info)

    async def stop(self):
        if self.aiozc:
            print(f"🛑 [Bonjour] Stopping advertisement of {self.name}")
            if self.info:
                await self.aiozc.zeroconf.async_unregister_service(self.info)
            await self.aiozc.close()
            self.aiozc = None
            self.info = None
