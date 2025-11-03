package com.example.api.repository;

import com.example.api.model.Notification;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface NotificationRepository extends JpaRepository<Notification, Long> {
    List<Notification> findByUserId(Long userId);
    List<Notification> findByUserIdAndIsRead(Long userId, Boolean isRead);
    List<Notification> findByType(Notification.NotificationType type);
    long countByUserIdAndIsRead(Long userId, Boolean isRead);
}
