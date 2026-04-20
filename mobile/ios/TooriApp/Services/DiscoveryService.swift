import Foundation
import Network
import Combine

final class DiscoveryService: ObservableObject {
    @Published var discoveredEndpoint: URL?
    
    private var browser: NWBrowser?
    private let serviceType = "_toori._tcp"
    
    func startBrowsing() {
        print("🔍 [Discovery] Starting browser for \(serviceType)...")
        let parameters = NWParameters()
        parameters.includePeerToPeer = true
        
        browser = NWBrowser(for: .bonjour(type: serviceType, domain: nil), using: .tcp)
        
        browser?.browseResultsChangedHandler = { [weak self] results, _ in
            guard let self = self else { return }
            
            // Pick the first viable result
            if let result = results.first {
                self.resolve(result: result)
            } else {
                DispatchQueue.main.async {
                    self.discoveredEndpoint = nil
                }
            }
        }
        
        browser?.start(queue: .main)
    }
    
    private func resolve(result: NWBrowser.Result) {
        if case let .service(name, _, _, _) = result.endpoint {
            // Standard Bonjour host resolution: <name>.local
            let host = "\(name).local"
            // Default port fallback to 7777 if not found, though Bonjour discovery 
            // will standardly provide the right port when resolving.
            let port = 7777 
            
            let urlString = "http://\(host):\(port)"
            if let url = URL(string: urlString) {
                print("✅ [Discovery] Found backend at \(url)")
                DispatchQueue.main.async {
                    self.discoveredEndpoint = url
                }
            }
        }
    }
    
    func stopBrowsing() {
        browser?.cancel()
        browser = nil
    }
}
