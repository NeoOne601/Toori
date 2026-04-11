import Foundation
#if canImport(BackgroundTasks)
import BackgroundTasks
#endif

public class SilentJournalEngine {
    
    private let gemma = GemmaModelManager.shared
    private let storageKey = "smriti.journal."
    
    public init() {}
    
    public func generateTodaysJournal() async throws -> String {
        let api = SmritiAPI()
        let observations = try await api.listObservations(limit: 50, summaryOnly: true)
        
        let threshold = Date().addingTimeInterval(-86400)
        let recentItems = observations.observations.filter { $0.created_at >= threshold }
        
        let highSurpriseItems = recentItems.filter { $0.effectiveSurpriseScore > 0.65 }
        let topItems = Array(highSurpriseItems.prefix(10))
        
        struct SummaryItem: Encodable { let summary: String; let surprise: Double }
        let summaryObjects = topItems.compactMap { item -> SummaryItem? in
            guard let summary = item.summary else { return nil }
            return SummaryItem(summary: summary, surprise: item.effectiveSurpriseScore)
        }
        
        let jsonSummary = String(data: (try? JSONEncoder().encode(summaryObjects)) ?? Data(), encoding: .utf8) ?? "[]"
        
        let entry: String
        let available = await gemma.isAvailable()
        if available {
            let displayLang = Locale.current.localizedString(
                forLanguageCode: Locale.current.language.languageCode?.identifier ?? "en"
            ) ?? "English"
            
            let prompt = """
            You are a private memory journal. Write exactly one paragraph
            (4-6 sentences) about today based only on these observations.
            Focus on: what was surprising, what was routine, what changed.
            Write in \(displayLang). Do not mention scores, identifiers,
            or technical terms. Write warmly, as if remembering a day.
            Observations: \(jsonSummary)
            """
            
            do {
                entry = try await gemma.generate(prompt: prompt, maxTokens: 200)
            } catch {
                entry = fallbackEntry(recentCount: recentItems.count, surpriseCount: highSurpriseItems.count)
            }
        } else {
            entry = fallbackEntry(recentCount: recentItems.count, surpriseCount: highSurpriseItems.count)
        }
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd"
        let dateStr = dateFormatter.string(from: Date())
        
        UserDefaults.standard.set(entry, forKey: storageKey + dateStr)
        pruneOldEntries()
        return entry
    }
    
    private func fallbackEntry(recentCount: Int, surpriseCount: Int) -> String {
        return "Today: \(recentCount) memories captured. \(surpriseCount) surprising moments."
    }
    
    private func pruneOldEntries() {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd"
        let thresholdDate = Date().addingTimeInterval(-86400 * 90)
        guard let thresholdStr = Int(dateFormatter.string(from: thresholdDate)) else { return }
        
        for key in UserDefaults.standard.dictionaryRepresentation().keys where key.hasPrefix(storageKey) {
            let suffix = key.dropFirst(storageKey.count)
            if let dateInt = Int(suffix), dateInt < thresholdStr {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }
    }
    
    public func cachedJournal(for date: Date) -> String? {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd"
        return UserDefaults.standard.string(forKey: storageKey + dateFormatter.string(from: date))
    }
    
    public func scheduleDaily() {
        #if os(macOS)
        let scheduler = NSBackgroundActivityScheduler(identifier: "com.toori.smriti.journal")
        scheduler.interval = 86400
        scheduler.tolerance = 3600
        scheduler.qualityOfService = .background
        scheduler.schedule { completion in
            if Calendar.current.component(.hour, from: Date()) >= 21 {
                Task {
                    _ = try? await self.generateTodaysJournal()
                    completion(.finished)
                }
            } else {
                completion(.deferred)
            }
        }
        #elseif os(iOS)
        let request = BGProcessingTaskRequest(identifier: "com.toori.smriti.journal")
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            // Silently ignore
        }
        #endif
    }
}
