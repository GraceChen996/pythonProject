for i in range(0, training_ebno_db.size):
    print('SNR={}'.format(training_ebno_db[i]))
    for it in range(0, train_iter):
        loss = 0
        with tf.GradientTape() as tape:
            c, x_hat, llr = model(distribution_num, training_ebno_db[i])  # 生成所有码字
            mask = np.ones(batch_size, dtype=bool)
            for j in range(0, distribution_num):

                if d_out == 0 or d_out >= d_in:
                    mask[j] = False
            # for j in range(0, c.shape[0]):
            #     d_in = sionna.utils.count_errors(sionna.utils.hard_decisions(llr[j]), c[j])
            #     d_out = sionna.utils.count_errors(sionna.utils.hard_decisions(x_hat[j]), c[j])
            #     if d_out == 0 or d_out >= d_in:
            #         mask[j] = False
            c = tf.boolean_mask(c, mask)
            x_hat = tf.boolean_mask(x_hat, mask)
            llr = tf.boolean_mask(llr, mask)

            # --- implement multi-loss as proposed by Nachmani et al. [1]---
            for ind in range(num_iter):
                # temp = tf.cast(sionna.utils.hard_decisions(x_hat), tf.float32)
                loss += bce(c, x_hat)  # add loss after each iteration
            loss /= num_iter  # scale loss by number of iterations

        grads = tape.gradient(loss, model.trainable_variables)
        grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if it % 10 == 0:  # evaluate every 10 iterations
            ber = compute_ber(c, hard_decisions(x_hat))
            # and print results
            mi = bmi(c, llr).numpy()  # calculate bit-wise mutual information
            l = loss.numpy()  # copy loss to numpy for printing
            print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
            bmi.reset_states()  # reset the BMI metric
