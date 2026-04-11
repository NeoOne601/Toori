import Foundation
#if canImport(BackgroundTasks)
import BackgroundTasks
#endif

public class AnticipationEngine {
    
    private let gemma = GemmaModelManager.shared
    
    public init() {}
    
    public func generateWeeklyInsight() async throws {
        let api = SmritiAPI()
        let observations = try await api.listObservations(limit: 200, summaryOnly: true)
        
        let tags = observations.observations.flatMap { $0.tags }
        var counts: [String: Int] = [:]
        for tag in tags { counts[tag, default: 0] += 1 }
        
        let topBehavior = counts.sorted { $0.value > $1.value }.prefix(5).map { "\($0.key) (\($0.value) occurrences)" }.joined(separator: ", ")
        
        let insight: String
        if await gemma.isAvailable(), !topBehavior.isEmpty {
            let prompt = """
            You are a behavioral analyst memory engine. Based on these top tags over the past week: \(topBehavior), write one short insightful sentence about a habit or pattern you've noticed. Do not use robotic language. Be observant and gentle.
            """
            insight = (try? await gemma.generate(prompt: prompt, maxTokens: 80)) ?? "You've been engaging heavily with \(counts.keys.first ?? "various subjects") lately."
        } else {
            insight = "Your routines remain stable over the past week."
        }
        
        UserDefaults.standard.set(insight, forKey: "smriti.insight.latest")
        let cal = Calendar.current
        let week = cal.component(.weekOfYear, from: Date())
        let year = cal.component(.yearForWeekOfYear, from: Date())
        UserDefaults.standard.set("\(year)-W\(week)", forKey: "smriti.insight.week")
        UserDefaults.standard.set(false, forKey: "smriti.insight.dismissed")
    }
    
    public func scheduleWeekly() {
        #if os(macOS)
        let scheduler = NSBackgroundActivityScheduler(identifier: "com.toori.smriti.patterns")
        scheduler.interval = 86400 * 7 // Weekly
        scheduler.tolerance = 86400
        scheduler.qualityOfService = .background
        scheduler.schedule { completion in
            Task {
                _ = try? await self.generateWeeklyInsight()
                completion(.finished)
            }
        }
        #elseif os(iOS)
        let request = BGProcessingTaskRequest(identifier: "com.toori.smriti.patterns")
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
        }
        #endif
    }
}
