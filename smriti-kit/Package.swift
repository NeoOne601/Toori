// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "SmritiKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "SmritiKit", targets: ["SmritiKit"]),
    ],
    targets: [
        .target(
            name: "SmritiKit",
            path: "Sources/SmritiKit"
        ),
    ]
)
